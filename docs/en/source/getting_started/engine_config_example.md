# Engine Config Example

- [Format](md-format)
- [Detailed Description](md-detailed-description)

(md-format)=

## Format

```json
{
    "LOG_PATH": "H:/OpenXRLab/Log/",
    "LOG_LEVEL": 0,
    "HEADLESS_SIMULATION": false,
    "ASSET_DIRECTORY": "H:/OpenXRLab/Assets/"
}
```

(md-detailed-description)=

## Detailed Description

|PARAM|TYPE|DEFAULT VALUE|DESCRIPTION|
| ----- | ----- | ----- | ----- |
|LOG\_PATH|string|-|Where to save log files.|
|LOG\_LEVEL|int|0|Minimum level to be logged. 0: TRACE; 1: DEBUG; 2: INFO; 3: WARN; 4: ERROR|
|HEADLESS\_SIMULATION|bool|false|Whether render OpenGL window during simulation.|
|ASSET\_DIRECTORY|string|-|Path to assets directory.|