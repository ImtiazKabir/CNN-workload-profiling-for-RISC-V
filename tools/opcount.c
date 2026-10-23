/*
 * opcount: QEMU TCG plugin that counts executed guest instructions per
 * (mnemonic, function symbol) pair.
 *
 * Each distinct pair gets one u64 slot in a per-vCPU scoreboard and every
 * translated instruction gets an inline "add 1" on its slot, so there is no
 * per-instruction callback at run time.
 *
 * usage: qemu-riscv64 -plugin ./opcount.so,outfile=counts.csv prog args...
 * output: CSV "mnemonic,symbol,count" ("symbol" empty when unknown, e.g. libc).
 * 2-byte (RVC) instructions are reported as "c.<expanded mnemonic>".
 */
#include <glib.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

#define MAX_KEYS (1 << 16)

typedef struct {
    char *mnemonic;
    char *symbol;
} Key;

static GMutex lock;
static GHashTable *key_index; /* "mnemonic\tsymbol" -> slot + 1 */
static Key keys[MAX_KEYS];
static size_t n_keys;
static struct qemu_plugin_scoreboard *counts;
static char *outfile;

static size_t slot_for(const char *mnemonic, const char *symbol)
{
    g_autofree char *k = g_strdup_printf("%s\t%s", mnemonic, symbol);
    size_t slot = GPOINTER_TO_SIZE(g_hash_table_lookup(key_index, k));

    if (slot) {
        return slot - 1;
    }
    g_assert(n_keys < MAX_KEYS);
    slot = n_keys++;
    keys[slot].mnemonic = g_strdup(mnemonic);
    keys[slot].symbol = g_strdup(symbol);
    g_hash_table_insert(key_index, g_steal_pointer(&k), GSIZE_TO_POINTER(slot + 1));
    return slot;
}

static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    size_t n = qemu_plugin_tb_n_insns(tb);

    g_mutex_lock(&lock);
    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        g_autofree char *disas = qemu_plugin_insn_disas(insn);
        const char *symbol = qemu_plugin_insn_symbol(insn);
        /* QEMU prints RVC instructions by their expanded name; tag them "c." */
        const char *prefix = qemu_plugin_insn_size(insn) == 2 ? "c." : "";
        char mnemonic[64];
        size_t len = strcspn(disas, " \t");

        len = MIN(len, sizeof(mnemonic) - strlen(prefix) - 1);
        strcpy(mnemonic, prefix);
        strncat(mnemonic, disas, len);

        qemu_plugin_u64 entry = {
            .score = counts,
            .offset = slot_for(mnemonic, symbol ? symbol : "") * sizeof(uint64_t),
        };
        qemu_plugin_register_vcpu_insn_exec_inline_per_vcpu(
            insn, QEMU_PLUGIN_INLINE_ADD_U64, entry, 1);
    }
    g_mutex_unlock(&lock);
}

static void plugin_exit(qemu_plugin_id_t id, void *p)
{
    FILE *f = outfile ? fopen(outfile, "w") : stderr;

    if (!f) {
        perror(outfile);
        return;
    }
    fprintf(f, "mnemonic,symbol,count\n");
    for (size_t i = 0; i < n_keys; i++) {
        qemu_plugin_u64 entry = { .score = counts, .offset = i * sizeof(uint64_t) };
        uint64_t c = qemu_plugin_u64_sum(entry);

        if (c) {
            fprintf(f, "%s,%s,%" PRIu64 "\n", keys[i].mnemonic, keys[i].symbol, c);
        }
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
        } else {
            fprintf(stderr, "opcount: unknown option %s\n", argv[i]);
            return -1;
        }
    }
    key_index = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    counts = qemu_plugin_scoreboard_new(MAX_KEYS * sizeof(uint64_t));
    qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
    return 0;
}
