## 실행환경
| 항목       | 버전  |
|-----------|-------|
| Ubuntu     | 22.04.14 |
| CARLA    | 0.9.14 |
| ROS2      | Humble |
| Python    | 3.8   |

## LeGO-LOAM 모듈 실행
```bash
ros2 launch lego_loam_sr run.launch.py 
```

## CARLA 서버 실행
- First, Locate **CARLA** directory in carla0.9.14
```bash
cd carla0.9.14
./CarlaUE4.sh -quality-level=Low
```
**-quality-level=Low** is optional

## CARLA 클라이언트 실행
```bash
python CARLA/automatic.py
```
