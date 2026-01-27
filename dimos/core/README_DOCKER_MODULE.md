# Docker Module System

Run DimOS modules inside Docker containers. From the outside, they behave exactly like normal modules.

## Core Idea

```
┌─────────────┐      streams/RPC      ┌─────────────┐
│  Module A   │◄─────────────────────►│  Module B   │
│  (native)   │                       │  (Docker)   │
└─────────────┘                       └─────────────┘
```

A Docker module is just a module. It has inputs, outputs, and RPCs. Other modules connect to it the same way they connect to any module.

## What Happens

1. `DockerModule` starts a container running the actual module
2. Communication happens over LCM multicast
3. Streams and RPCs work transparently

## Dockerfile Conversion

Any Python Dockerfile becomes DimOS-ready by appending a small footer that installs DimOS and sets the entrypoint. The container receives module class and config as a JSON string on startup.

## Future work

Once the spec is approved for the modifiaciton in pLCM stream can make the changes in TransportSpec pattern adding it to_spec() on each transport class and adding from_spec() factory to reconstruct the transport, enabling SHM, JpegSHM, JpegLCM.