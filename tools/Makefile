QEMU_INC ?= ../third_party/qemu/include/qemu
CFLAGS ?= -O2 -Wall
PLUGIN_FLAGS = -shared -fPIC -I$(QEMU_INC) $(shell pkg-config --cflags glib-2.0)

all: opcount.so opcount_roi.so

%.so: %.c
	$(CC) $(CFLAGS) $(PLUGIN_FLAGS) -o $@ $<

clean:
	rm -f opcount.so opcount_roi.so
