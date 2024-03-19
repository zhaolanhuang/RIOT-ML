Network Setup on Local Boards
=====================

(Adapted from [SUIT update examples](https://github.com/RIOT-OS/RIOT/blob/master/examples/suit_update/README.hardware.md#Setup))

You may choose one of the two options to set up your local experimental (IPv6) IoT network:

- Option 1: [Set up a wired device using *Ethernet over Serial(ETHOS)*][setup-wired]
  - In this case a area network will directly set up between local workstation and IoT node with ML model.

- Option 2: [Setup a wireless device behind a border router][setup-wireless]
  - In this case a *wireless* area network will set up between local workstation and IoT node with ML model through a boarder router(another IoT board with wireless capability).

# Option 1: Setup a wired device using ETHOS
[setup-wired]: #Option-1-Setup-a-wired-device-using-ethos

## Configure the network
[setup-wired-network]: #Configure-the-network

In one terminal, start:

```
sudo RIOT/dist/tools/ethos/setup_network.sh riot0 2001:db8::/64
```

This will create a tap interface called `riot0`, owned by the user. It will
also run an instance of uhcpcd, which starts serving the prefix
`2001:db8::/64`.

From another terminal on the host, add a routable address on the host `riot0`
interface:

```
sudo ip address add 2001:db8::1/128 dev riot0
```

## Provision the device
[setup-wired-provision]: #Provision-the-device

First, plug the IoT board in the computer.

In order to get a RIOT-ML firmware onto the node, run

```
USE_ETHOS=1 python model_update.py --preprovision --board <IoT board name> <Path to model file>
```
e.g.

```
USE_ETHOS=1 python model_update.py --preprovision --board nrf52840dk ./model_zoo/mnist_0.983_quantized.tflite
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

Plug the BR board on the computer and flash the
[gnrc_border_router](https://github.com/RIOT-OS/RIOT/tree/master/examples/gnrc_border_router)
application on it:

```
make BOARD=<BR board> -C RIOT/examples/gnrc_border_router flash
```

You may specify 802.15.4 channel by set up `DEFAULT_CHANNEL=<channel number>`, e.g.,

```
DEFAULT_CHANNEL=26 make BOARD=<BR board> -C RIOT/examples/gnrc_border_router flash
```
The default channel is 26. Channel number should range from 11 to 26.

On terminal, start the network (assuming on the host the virtual port of the
board is `/dev/ttyACM0`):

```
sudo ./RIOT/dist/tools/ethos/start_network.sh /dev/ttyACM0 riot0 2001:db8::/64
```
Keep this terminal open. We name it as __Terminal BR__.

From another terminal on the host, add a routable address on the host `riot0`
interface:

```
sudo ip address add 2001:db9::1/128 dev riot0
```

## Provision the wireless device
[setup-wireless-provision]: #Provision-the-wireless-device

In this scenario the node will be connected through a border
router. Ethos must be disabled in the firmware when building and flashing the firmware:

```
USE_ETHOS=0 DEFAULT_CHANNEL=26 python model_update.py --preprovision --board nrf52840dk ./model_zoo/mnist_0.983_quantized.tflite
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

