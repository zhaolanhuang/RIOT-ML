# Required variables defined in riotboot.inc.mk or Makefile.include
BINDIR_APP = $(CURDIR)/bin/$(BOARD)/$(APPLICATION)
$(BINDIR_APP): $(CLEAN)
	$(Q)mkdir -p $(BINDIR_APP)

# Include to be able to use memoized
include $(RIOTBASE)/makefiles/utils/variables.mk
EPOCH = $(call memoized,EPOCH,$(shell date +%s))
APP_VER ?= $(EPOCH)

# Default addressing if following README.native.md
ifeq ($(BOARD),native)
  SUIT_CLIENT ?= [2001:db8::2]
  SUIT_COAP_SERVER ?= [2001:db8::1]
  $(call target-export-variables,test-with-config,SUIT_COAP_SERVER)
endif

ifeq ($(BOARD),native)
  # Set settings for publishing fake fw payloads to native
  SUIT_NATIVE_PAYLOAD ?= "AABBCCDD"
  SUIT_NATIVE_PAYLOAD_BIN ?= $(BINDIR_APP)/fw.$(APP_VER).bin
  # Make sure it is built
  BUILD_FILES += $(SUIT_NATIVE_PAYLOAD_BIN)

  $(SUIT_NATIVE_PAYLOAD_BIN): $(BINDIR_APP)
		$(Q)echo $(SUIT_NATIVE_PAYLOAD) > $@

  SUIT_FW_STORAGE ?= /nvm0/SLOT0.TXT
  SUIT_MANIFEST_PAYLOADS ?= $(SUIT_NATIVE_PAYLOAD_BIN)
  SUIT_MANIFEST_SLOTFILES ?= $(SUIT_NATIVE_PAYLOAD_BIN):0:$(SUIT_FW_STORAGE)
endif

suit/notify/model_params: | $(filter suit/publish, $(MAKECMDGOALS))
	$(Q)test -n "$(SUIT_CLIENT)" || { echo "error: SUIT_CLIENT unset!"; false; }
	aiocoap-client -m POST "coap://$(SUIT_CLIENT)/model/params/update" \
		--payload "$(SUIT_COAP_ROOT)/$(SUIT_NOTIFY_MANIFEST)" && \
		echo "Triggered $(SUIT_CLIENT) to update."