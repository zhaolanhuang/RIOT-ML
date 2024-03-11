Running Model Update on IoT Boards
=====

# Prequisites

Please set up your area network and preprovison SUIT capable firmware for IoT boards before conducting model update. If you're using local boards, please check [Network Setup on Local Boards](https://github.com/zhaolanhuang/RIOT-ML/blob/main/examples/Network_Setup_Local.md); If you consider remote boards on FIT IoT-LAB, please check [Network Setup on Remote Boards from FIT IoT-LAB](https://github.com/zhaolanhuang/RIOT-ML/blob/main/examples/Network_Setup_IOTLAB.md). Also please keep IPv6 address of RIOT-ML node in record.

After setup of IoT area network, open new terminal and start CoAP file server as artifacts repository.

```
cd coaproot
aiocoap-fileserver .
```
Keep the server running in the terminal.

# Full Update
The following command executes full model update on RIOT-ML node:

```
USE_ETHOS=0 python model_update.py --full --board nrf52840dk --client <ip of RIOT-ML node> --server <ip of local computer> <path to the model file>
```

For example,

```
USE_ETHOS=0 python model_update.py --full-update --board nrf52840dk --client [2001:db8::64fa:5ffe:7879:4ad9] --server [2001:db8::1] ./model_zoo/mnist_0.983_quantized.tflite
```

# Partial Update
The following command executes partial model update on RIOT-ML node:

```
USE_ETHOS=0 python model_update.py --full --board nrf52840dk --client <ip of RIOT-ML node> --server <ip of local computer> <path to the model parameter file in JSON format>
```

For example,

```
USE_ETHOS=0 python model_update.py --partial-update --board nrf52840dk --client [2001:db8::64fa:5ffe:7879:4ad9] --server [2001:db8::1] ./model_zoo/model_params_lenet5.json
```
The parameter file shoud be like
```
{
    '<parameter1 name>' : <list of new parameter values>,
    '<parameter2 name>' : <list of new parameter values>,
    ...
}
```