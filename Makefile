RIOTBASE= ./RIOT

BOARD ?= stm32f746g-disco
APPLICATION = U-TOE

# WERROR ?= 0
# DEVELHELP ?= 1

CFLAGS += -fstack-usage
CFLAGS += -fcallgraph-info
CFLAGS += -fdump-rtl-expand
CFLAGS += -fdump-tree-optimized

EXTERNAL_PKG_DIRS += models

USEPKG += default 
USEMODULE += xtimer random stdin

UTOE_RANDOM_SEED ?= 42
UTOE_TRIAL_NUM ?= 10
UTOE_GRANULARITY ?= 0

CFLAGS += -DUTOE_RANDOM_SEED=$(UTOE_RANDOM_SEED) -DUTOE_TRIAL_NUM=$(UTOE_TRIAL_NUM)
CFLAGS += -DUTOE_GRANULARITY=$(UTOE_GRANULARITY) -DCONFIG_SKIP_BOOT_MSG=1

INCLUDES += -I$(CURDIR)/utvm_runtime/include

ifeq ($(UTOE_GRANULARITY), 1)

CFLAGS += -DTHREAD_STACKSIZE_DEFAULT=2048
EXTERNAL_PKG_DIRS += $(CURDIR)
USEPKG += utvm_runtime

endif

USEMODULE += mlmci
USEMODULE += mlmodel_coap
EXTERNAL_MODULE_DIRS += $(CURDIR)

USE_SUIT ?= 0
SUIT_COAP_FSROOT ?= $(CURDIR)/coaproot
ifeq ($(USE_SUIT), 1)
	include $(CURDIR)/suit.mk
	DIRS += suit_helpers
	USEMODULE += suit_helpers
	CFLAGS += -DUSE_SUIT
else
	include $(RIOTBASE)/Makefile.include
endif

CFLAGS += -Wno-strict-prototypes 
CFLAGS += -Wno-missing-include-dirs

IOTLAB_ARCHI_openmote-b = openmoteb
include iotlab.site.mk
include $(RIOTBASE)/dist/testbed-support/makefile.iotlab.archi.inc.mk
include $(RIOTBASE)/dist/testbed-support/Makefile.iotlab
override BINARY := $(ELFFILE)

list-ttys-json:
	$(Q) python $(RIOTTOOLS)/usb-serial/ttys.py --format json

