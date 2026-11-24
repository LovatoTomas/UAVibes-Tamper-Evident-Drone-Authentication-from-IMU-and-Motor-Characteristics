# Host build of the on-board core (Linux/WSL gcc). Used by
# auth_client/tests/test_onboard_c.py; also: make -C firmware/uavibe_key/test
CC ?= gcc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter
SRC = ../src/core/uv_features.c ../src/core/uv_fe.c ../src/core/uv_crypto.c \
      ../src/vendor/monocypher/monocypher.c ../src/vendor/monocypher/monocypher-ed25519.c host_test.c
INC = -I../src/core -I../src/vendor/monocypher

host_test: $(SRC) ../src/core/uv_core.h ../src/core/uv_sysparams.h
	$(CC) $(CFLAGS) $(INC) $(SRC) -lm -o $@

clean:
	rm -f host_test
