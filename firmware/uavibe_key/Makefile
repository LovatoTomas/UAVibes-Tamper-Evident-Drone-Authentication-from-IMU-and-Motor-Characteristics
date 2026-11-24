# UAVibes key firmware (variant FW2): Crazyflie firmware + FW1 board patch +
# this app. Out-of-tree build against the sibling crazyflie-firmware checkout
# (which must carry the FW1 patch, firmware/patches/fw1-board-selftest.patch).
#
#   make            -> build/cf2.bin        (in WSL: sudo apt install make gcc-arm-none-eabi)
#   make cload      -> flash over the radio (drone in bootloader mode, only it powered)
#
# The FW1 firmware itself is built in the crazyflie-firmware folder as before
# (make; make cload). Switching variant = flashing the other cf2.bin.
CRAZYFLIE_BASE ?= ../../../crazyflie-firmware
OOT_CONFIG := $(PWD)/app-config
EXTRA_CFLAGS += -I$(PWD)/src/core -I$(PWD)/src/vendor/monocypher

# UV_RESEARCH=1 (default): build for the lab. Keeps the read-only param group
# "uavibe" (board fingerprint, read by the dashboard and the bench script) and
# the GET/PUT_HELPER test messages of the cloning experiment.
# UV_RESEARCH=0: build for a drone in service. Leaves both out: the board values
# are part of the reading the key comes from, and params can be read by anyone
# in radio range.   make clean && make UV_RESEARCH=0
UV_RESEARCH ?= 1
ifeq ($(UV_RESEARCH),0)
EXTRA_CFLAGS += -DUV_HIDE_BOARD_PARAMS -DUV_SECURITY_TESTS=0
endif

include $(CRAZYFLIE_BASE)/tools/make/oot.mk
