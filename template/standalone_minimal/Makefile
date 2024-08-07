RIOTBASE= ../RIOT-ML/RIOT

BOARD ?= ${board}
APPLICATION = RIOT-ML_standalone_minimal

EXTERNAL_PKG_DIRS += models

USEPKG += default 
USEMODULE += xtimer random stdin

USEMODULE += mlmci
EXTERNAL_MODULE_DIRS += $(CURDIR)

CFLAGS += -Wno-strict-prototypes 
# CFLAGS += -Wno-missing-include-dirs
CFLAGS += -Wno-discarded-qualifiers

include $(RIOTBASE)/Makefile.include