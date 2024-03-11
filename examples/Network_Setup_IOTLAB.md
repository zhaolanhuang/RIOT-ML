Network Setup on Remote Boards from FIT IoT-LAB
=====================

- Prequisites
Please set up your IoT-LAB account and credential for access to remote boards (see [FIT IoT-LAB Setup](https://github.com/zhaolanhuang/RIOT-ML#optional-fit-iot-lab)).


This setup is based on two remote boards in FIT IoT-LAB: iotlab-m3 as boarder router, and nrf52840dk as RIOT-ML node. You don't need any IoT boards in hand. 

You may choose one of the two options to set up your local experimental (IPv6) IoT network:

- Option 1: [Set up a wired device using *Ethernet over Serial(ETHOS)*][setup-wired]
  - In this case a area network will directly set up between local workstation and remote IoT node with ML model.

- Option 2: [Setup a wireless device behind a border router][setup-wireless]
  - In this case a *wireless* area network will set up between local workstation and remote IoT node with ML model through a remote boarder router(another IoT board with wireless capability).

(Belows were adapted from [SUIT update examples](https://github.com/RIOT-OS/RIOT/blob/master/examples/suit_update/README.hardware.md#Setup))

# Option 1: Setup a wired device using ETHOS
[setup-wired]: #Option-1-Setup-a-wired-device-using-ethos

## Submit an IoT-LAB experiment
In order to activate the remote boards we need to submit an IoT-LAB experiment, e.g.,

```
iotlab-experiment submit -n riot_ml -l 1,archi=nrf52840dk:multi+site=saclay -d 60
```
This command requests one board (nrf52840dk) in site Saclay,France for 60 minutes. After a while we can use
```
iotlab-experiment get -n
``` 
to gather the information of allocated board. You may see outputs like:
```
{
    "items": [
        {
            # As RIOT-ML node
            "archi": "nrf52840dk:multi",
            ...
            "network_address": "nrf52840dk-10.saclay.iot-lab.info",
            ...
        }
    ]
}
```
Please keep `network_address` of each board for further steps.

## Setup SSH tunnel to node
Open another terminal and run
```
ssh -L 20000:<network address to boarder router>:20000 <username of iotlab>@saclay.iot-lab.info
```
In this case it should be
```
ssh -L 20000:nrf52840dk-10.saclay.iot-lab.info:20000 <username of iotlab>@saclay.iot-lab.info
```
Keep this terminal open.

## Configure the network
[setup-wired-network]: #Configure-the-network

In one terminal, start:

```
sudo RIOT/dist/tools/ethos/start_network.sh tcp:127.0.0.1 riot0 2001:db8::/64 20000
```

This will create a tap interface called `riot0`, owned by the user. It will
also run an instance of uhcpcd, which starts serving the prefix
`2001:db8::/64`. Keep the shell open as long as you need the network.

## Provision the device
[setup-wired-provision]: #Provision-the-device

First, plug the IoT board in the computer.

In order to get a RIOT-ML firmware onto the node, run

```
USE_ETHOS=1 python model_update.py --preprovision --board <IoT board name> --iotlab-node <network address to RIOT-ML node> <Path to model file>
```
e.g.

```
USE_ETHOS=1 python model_update.py --preprovision --board nrf52840dk --iotlab-node nrf52840dk-10.saclay.iot-lab.info ./model_zoo/mnist_0.983_quantized.tflite
```
This command compiles and flashes a RIOT-ML firmware with pre-trained LeNet5 model onto the nrf52840dk board.

This command also generates the cryptographic keys (private/public) used to
sign and verify the manifest and images. See the "Key generation" section in
[SUIT detailed explanation](https://github.com/RIOT-OS/RIOT/blob/master/examples/suit_update/README.hardware.md#key-generation) for details.

From another terminal on the host, add a routable address on the host `riot0`
interface:

```
sudo ip address add 2001:db8::1/128 dev riot0
```

In another terminal, run:

```
ip -6 neighbour
```

You may see the IP address of the linked board, i.e.,
```
2001:db8::64fa:5ffe:7879:4ad9 dev riot0 STALE
```
Please keep that in record for further experiments.

# Option 2: Setup a wireless device behind a border router
[setup-wireless]: #Option-2-Setup-a-wireless-device-behind-a-border-router

If the workflow for updating using ethos is successful, you can try doing the
same over wireless network interfaces, by updating a node that is connected
wirelessly with a border router in between.

Depending on your device you can use BLE or 802.15.4.

## Submit an IoT-LAB experiment
In order to activate the remote boards we need to submit an IoT-LAB experiment, e.g.,

```
iotlab-experiment submit -n riot_ml -l 1,archi=m3:at86rf231+site=saclay -l 1,archi=nrf52840dk:multi+site=saclay -d 60
```
This command requests two boards (iotlab-m3 and nrf52840dk) in site Saclay,France for 60 minutes. After a while we can use
```
iotlab-experiment get -n
``` 
to gather the information of allocated boards. You may see outputs like:
```
{
    "items": [
        {
            # As boarder router
            "archi": "m3:at86rf231", 
            ...
            "network_address": "m3-10.saclay.iot-lab.info",
            ...
        },
        {
            # As RIOT-ML node
            "archi": "nrf52840dk:multi",
            ...
            "network_address": "nrf52840dk-10.saclay.iot-lab.info",
            ...
        }
    ]
}
```
Please keep `network_address` of each board for further steps.

## Setup SSH tunnel to boarder router
Open another terminal and run
```
ssh -L 20000:<network address to boarder router>:20000 <username of iotlab>@saclay.iot-lab.info
```
In this case it should be
```
ssh -L 20000:m3-10.saclay.iot-lab.info:20000 <username of iotlab>@saclay.iot-lab.info
```
Keep this terminal open.

## Configure the wireless network

[setup-wireless-network]: #Configure-the-wireless-network

A wireless node has no direct connection to the Internet so a border router (BR)
between 802.15.4/BLE and Ethernet must be configured.
Any board providing a 802.15.4/BLE radio can be used as BR.

If configuring a BLE network when flashing the device include
`USEMODULE+=nimble_autoconn_ipsp` in the application Makefile, or prefix all
your make commands with it (for the BR as well as the device), e.g.:

```
USEMODULE+=nimble_autoconn_ipsp make BOARD=<BR board>
```

Flash the
[gnrc_border_router](https://github.com/RIOT-OS/RIOT/tree/master/examples/gnrc_border_router)
application on boarder router:

```
IOTLAB_NODE=m3-10.saclay.iot-lab.info ETHOS_BAUDRATE=500000 BOARD=iotlab-m3 make -C RIOT/examples/gnrc_border_router flash
```

You may specify 802.15.4 channel by set up `DEFAULT_CHANNEL=<channel number>`, e.g.,

```
IOTLAB_NODE=m3-10.saclay.iot-lab.info ETHOS_BAUDRATE=500000 DEFAULT_CHANNEL=26 BOARD=iotlab-m3 make -C RIOT/examples/gnrc_border_router flash
```
The default channel is 26. Channel number should range from 11 to 26.

On terminal, start the network (assuming on the host the virtual port of the
board is `tcp::127.0.0.1:2000`, which is forwarded to the serial port of remote boarder router):

```
sudo RIOT/dist/tools/ethos/start_network.sh tcp:127.0.0.1 riot0 2001:db8::/64 20000
```
Keep this terminal open. We name it as __Terminal BR__.

From another terminal on the host, add a routable address on the host `riot0`
interface:

```
sudo ip address add 2001:db8::1/128 dev riot0
```

## Provision the wireless device
[setup-wireless-provision]: #Provision-the-wireless-device

In this scenario the node will be connected through a border
router. Ethos must be disabled in the firmware when building and flashing the firmware:

```
USE_ETHOS=0 DEFAULT_CHANNEL=26 python model_update.py --preprovision --board nrf52840dk --iotlab-node <network address to RIOT-ML node> ./model_zoo/mnist_0.983_quantized.tflite
```
In this case it should be
```
USE_ETHOS=0 DEFAULT_CHANNEL=26 python model_update.py --preprovision --board nrf52840dk --iotlab-node nrf52840dk-10.saclay.iot-lab.info ./model_zoo/mnist_0.983_quantized.tflite
```

Switch back to the __Terminal BR__, we can get the global address of the node from border router via:

```
nib neigh
```
You may see the output as:

```
fe80::1 dev #6 lladdr 5E:56:EA:DA:D9:67  REACHABLE GC
fe80::5c56:eaff:feda:d967 dev #6 lladdr 5E:56:EA:DA:D9:67  STALE GC
2001:db8::64fa:5ffe:7879:4ad9 dev #5 lladdr 66:FA:5F:FE:78:79:4A:D9  REACHABLE REGISTERED
```

Here the global IPv6 is `2001:db8::64fa:5ffe:7879:4ad9`. Please keep that in record for further experiments.
**The address will be different according to your device and the chosen prefix**.
In this case the RIOT node can be reached from the host using its global address:

```
ping 2001:db8::64fa:5ffe:7879:4ad9
```

_NOTE_: when using BLE the connection might take a little longer, and you might not
see the global address right away. But the global address will always consist of the
the prefix (`2001:db8::`) and the EUI64 suffix, in this case `64fa:5ffe:7879:4ad9`.

