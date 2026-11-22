# Oval Warden

A Python bot for Builders Buildathon's Mech Royale track. It speaks `mechfight/1` over JSON lines and uses only Python's standard library.

## Bot approach

- Stay inside the current zone and relocate toward the next one before it closes.
- Move between safe waypoints, avoid nearby enemies and incoming projectiles, and use buildings as navigation obstacles.
- Collect nearby ground cubes and useful weapon drops when the route is safe.
- Favor defensive and mobility upgrades, then use predictive aim and weapon-specific firing logic.

## Run the conformance check

From this repository root:

```sh
python3 check/check.py -- python3 -u bot.py
```

Build and check the same sandbox used by the scorer:

```sh
docker build --platform linux/amd64 -t oval-warden .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" oval-warden
```

The checker verifies the protocol and response deadline. It is not a match simulator, so the practice score is the measure of strategy quality.
