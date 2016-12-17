The example does not run properly. It starts but no key events are sent to the docker instance (I guess).

I fixed some of the issues that have been mentioned [here](https://github.com/openai/universe/issues/6).

The only way I could see the game being started is by following these steps:

Docker:
``sudo docker run -p 5900:5900 -p 15900:15900 --privileged --cap-add SYS_ADMIN --ipc host quay.io/openai/universe.flashgames:0.20.7``

Followed by running ``example/recorders/reward_recorder.py``

Lastly running ``demo.py``
