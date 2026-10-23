/*
 * opcount_roi: QEMU TCG plugin that counts executed guest instructions per
 * (mapped file, mnemonic), for the whole run and for a region of interest.
 *
 * Every translated block gets one u64 slot in a per-vCPU scoreboard with an
 * inline "add 1" on block entry; at exit each instruction of a block is
 * credited with the block's count.
 *
 * Region of interest: the guest marks it with getppid(MAGIC, 1) / getppid(MAGIC, 2)
 * (getppid ignores its arguments, so this is harmless natively). The block
 * counters are snapshotted at each marker and (end - begin) is accumulated, so
 * several begin/end pairs add up.
 *
 * Library attribution: guest addresses are mapped to files with a copy of the
 * guest's /proc/self/maps, written by the guest (see tools/riscv_roi.py) and
 * read at exit.
 *
 * usage: qemu-riscv64 -plugin ./opcount_roi.so,outfile=out.csv,maps=maps.txt prog
 * output: CSV "scope,object,mnemonic,count", scope = all | roi.
 * 2-byte (RVC) instructions are reported as "c.<expanded mnemonic>".
 */
#include <glib.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

#define ROI_MAGIC 0x524f49 /* "ROI" */
#define SYS_GETPPID 173    /* riscv64 */

typedef struct {
    size_t n;
    uint64_t *pcs;
    uint32_t *mnemonics; /* index into mnemonic_names */
} Block;

typedef struct {
    uint64_t start, end;
    char *path;
} Mapping;

static GMutex lock;
static GPtrArray *blocks;         /* Block*, index = scoreboard slot */
static GHashTable *mnemonic_index; /* name -> id + 1 */
static GPtrArray *mnemonic_names;
static struct qemu_plugin_scoreboard *counts;
static size_t max_blocks = 1 << 22;
static uint64_t *roi_begin_snapshot; /* NULL when no ROI is open */
static uint64_t *roi_counts;
static char *outfile, *mapsfile;

static uint64_t block_count(size_t slot)
{
    qemu_plugin_u64 entry = { .score = counts, .offset = slot * sizeof(uint64_t) };
    return qemu_plugin_u64_sum(entry);
}

static uint32_t mnemonic_id(const char *name)
{
    size_t id = GPOINTER_TO_SIZE(g_hash_table_lookup(mnemonic_index, name));

    if (id) {
        return id - 1;
    }
    id = mnemonic_names->len;
    g_ptr_array_add(mnemonic_names, g_strdup(name));
    g_hash_table_insert(mnemonic_index, g_strdup(name), GSIZE_TO_POINTER(id + 1));
    return id;
}

static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    Block *b = g_new0(Block, 1);

    b->n = qemu_plugin_tb_n_insns(tb);
    b->pcs = g_new(uint64_t, b->n);
    b->mnemonics = g_new(uint32_t, b->n);

    g_mutex_lock(&lock);
    for (size_t i = 0; i < b->n; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        g_autofree char *disas = qemu_plugin_insn_disas(insn);
        const char *prefix = qemu_plugin_insn_size(insn) == 2 ? "c." : "";
        char mnemonic[64];
        size_t len = strcspn(disas, " \t");

        len = MIN(len, sizeof(mnemonic) - strlen(prefix) - 1);
        strcpy(mnemonic, prefix);
        strncat(mnemonic, disas, len);
        b->pcs[i] = qemu_plugin_insn_vaddr(insn);
        b->mnemonics[i] = mnemonic_id(mnemonic);
    }
    if (blocks->len >= max_blocks) {
        fprintf(stderr, "opcount_roi: more than %zu blocks, raise max_blocks=\n", max_blocks);
        abort();
    }
    qemu_plugin_u64 entry = {
        .score = counts, .offset = blocks->len * sizeof(uint64_t),
    };
    g_ptr_array_add(blocks, b);
    g_mutex_unlock(&lock);

    qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
        tb, QEMU_PLUGIN_INLINE_ADD_U64, entry, 1);
}

static void vcpu_syscall(qemu_plugin_id_t id, unsigned int vcpu_index,
                         int64_t num, uint64_t a1, uint64_t a2, uint64_t a3,
                         uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7,
                         uint64_t a8)
{
    if (num != SYS_GETPPID || a1 != ROI_MAGIC) {
        return;
    }
    g_mutex_lock(&lock);
    if (a2 == 1 && !roi_begin_snapshot) {
        roi_begin_snapshot = g_new(uint64_t, blocks->len + 1);
        roi_begin_snapshot[0] = blocks->len;
        for (size_t i = 0; i < blocks->len; i++) {
            roi_begin_snapshot[i + 1] = block_count(i);
        }
    } else if (a2 == 2 && roi_begin_snapshot) {
        size_t n_before = roi_begin_snapshot[0];

        if (!roi_counts) {
            roi_counts = g_new0(uint64_t, max_blocks);
        }
        for (size_t i = 0; i < blocks->len; i++) {
            uint64_t before = i < n_before ? roi_begin_snapshot[i + 1] : 0;
            roi_counts[i] += block_count(i) - before;
        }
        g_free(roi_begin_snapshot);
        roi_begin_snapshot = NULL;
    }
    g_mutex_unlock(&lock);
}

static GArray *read_maps(void)
{
    GArray *maps = g_array_new(FALSE, FALSE, sizeof(Mapping));
    g_autofree char *text = NULL;

    if (!mapsfile || !g_file_get_contents(mapsfile, &text, NULL, NULL)) {
        return maps;
    }
    g_auto(GStrv) lines = g_strsplit(text, "\n", -1);
    for (char **l = lines; *l; l++) {
        Mapping m;
        char path[4096] = "";

        if (sscanf(*l, "%" SCNx64 "-%" SCNx64 " %*s %*s %*s %*s %4095[^\n]",
                   &m.start, &m.end, path) < 2) {
            continue;
        }
        m.path = g_strdup(path[0] ? path : "[anon]");
        g_array_append_val(maps, m);
    }
    return maps;
}

static const char *object_for(GArray *maps, uint64_t pc)
{
    size_t lo = 0, hi = maps->len;

    while (lo < hi) { /* maps are sorted by address */
        size_t mid = (lo + hi) / 2;
        Mapping *m = &g_array_index(maps, Mapping, mid);

        if (pc < m->start) {
            hi = mid;
        } else if (pc >= m->end) {
            lo = mid + 1;
        } else {
            return m->path;
        }
    }
    return "?";
}

static void add_count(GHashTable *agg, const char *scope, const char *object,
                      const char *mnemonic, uint64_t c)
{
    char *key = g_strdup_printf("%s,%s,%s", scope, object, mnemonic);
    uint64_t *v = g_hash_table_lookup(agg, key);

    if (!v) {
        v = g_new0(uint64_t, 1);
        g_hash_table_insert(agg, key, v);
    } else {
        g_free(key);
    }
    *v += c;
}

static void plugin_exit(qemu_plugin_id_t id, void *p)
{
    GArray *maps = read_maps();
    GHashTable *agg = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
    FILE *f = outfile ? fopen(outfile, "w") : stderr;

    if (!f) {
        perror(outfile);
        return;
    }
    for (size_t i = 0; i < blocks->len; i++) {
        Block *b = g_ptr_array_index(blocks, i);
        uint64_t all = block_count(i);
        uint64_t roi = roi_counts ? roi_counts[i] : 0;

        for (size_t j = 0; all && j < b->n; j++) {
            const char *object = object_for(maps, b->pcs[j]);
            const char *mnemonic = g_ptr_array_index(mnemonic_names, b->mnemonics[j]);

            add_count(agg, "all", object, mnemonic, all);
            if (roi) {
                add_count(agg, "roi", object, mnemonic, roi);
            }
        }
    }
    fprintf(f, "scope,object,mnemonic,count\n");
    GHashTableIter it;
    gpointer key, value;
    g_hash_table_iter_init(&it, agg);
    while (g_hash_table_iter_next(&it, &key, &value)) {
        fprintf(f, "%s,%" PRIu64 "\n", (char *)key, *(uint64_t *)value);
    }
    if (f != stderr) {
        fclose(f);
    }
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info,
                                           int argc, char **argv)
{
    for (int i = 0; i < argc; i++) {
        if (g_str_has_prefix(argv[i], "outfile=")) {
            outfile = g_strdup(argv[i] + strlen("outfile="));
        } else if (g_str_has_prefix(argv[i], "maps=")) {
            mapsfile = g_strdup(argv[i] + strlen("maps="));
        } else if (g_str_has_prefix(argv[i], "max_blocks=")) {
            max_blocks = strtoull(argv[i] + strlen("max_blocks="), NULL, 0);
        } else {
            fprintf(stderr, "opcount_roi: unknown option %s\n", argv[i]);
            return -1;
        }
    }
    blocks = g_ptr_array_new();
    mnemonic_index = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    mnemonic_names = g_ptr_array_new();
    counts = qemu_plugin_scoreboard_new(max_blocks * sizeof(uint64_t));
    qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);
    qemu_plugin_register_vcpu_syscall_cb(id, vcpu_syscall);
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
    return 0;
}
