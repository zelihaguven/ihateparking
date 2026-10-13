# Footage research for the English video-assist prototype

## Selected local asset

The first local demo asset is **A woman driving a car in a parking lot** by Magda Ehlers on Pexels. The page describes it as a POV/in-cabin driving clip and lists 3840×2160, 16:9, 13 seconds and 25 FPS. The downloaded demo copy is the 2560×1440 MP4 variant used by the prototype. [1]

Pexels states that its photos and videos can be downloaded and used for free, that attribution is not required, and that the media may be modified. It also prohibits selling unaltered copies, implying endorsement, or redistributing the media on stock platforms. We keep the source attribution in the project metadata and use the clip only as a transformed research/demo input. [2]

## What the clip actually supports

The local video analysis found:

- 00:00–00:06: forward driving on a cone course with visible orange boundaries;
- 00:06–00:10: a right-turning manoeuvre with active steering;
- 00:11–00:12: steering straightening followed by a stop;
- no complete parking slot and no reliable reverse manoeuvre in the visible camera view.

Therefore, v1 is explicitly called **Video Assist**, not a complete autonomous parking system. The implemented perception module detects orange boundary/cone responses and maps them to English human-in-the-loop suggestions such as `KEEP CENTERED`, `STEER RIGHT GENTLY`, `STEER LEFT GENTLY`, `SLOW DOWN`, and `STOP`. It does not claim to recover a real steering angle or to actuate a vehicle.

## Candidate sources considered

The YouTube video **Parallel Parking – The Reverse Camera Technique** contains useful reverse-camera segments, including actual reversing and forward correction. However, the footage is edited, includes camera transitions and overlays, and is not a stable raw sensor feed. It is useful as a visual reference, but it is not bundled or redistributed with this repository. [3]

The Dragon Lake Parking dataset is much more relevant scientifically: it contains 3.5 hours of 4K aerial video, parking-lot trajectories and many vehicle parking manoeuvres. Its raw video and annotations total 168 GB, require an access request, and are restricted to non-commercial use without redistribution. It is a strong future benchmark, but it is not a front-camera clip for the first browser demo. [4]

The E2E Parking Dataset is an open CARLA-based project with multi-camera RGB inputs, target-slot information and control signals. Its repository provides a dataset download link, generation/evaluation code and an MIT-licensed code repository. It is the right next step for repeatable simulated training and closed-loop evaluation, while the Pexels clip gives the current demo a real-world in-cabin appearance. [5] [6]

## Next footage requirement

To demonstrate the requested “stop / turn the wheel / reverse / straighten” assistant faithfully, the next data acquisition target is a continuous, fixed-camera clip with:

1. a visible parking slot or curb,
2. a forward approach,
3. a complete reverse manoeuvre,
4. no editing cuts, and
5. ideally a rear or 360-degree camera view.

The system should then run the same instruction policy against actual slot geometry rather than cone boundaries.

## References

[1]: https://www.pexels.com/video/a-woman-driving-a-car-in-a-parking-lot-18495959/ "A woman driving a car in a parking lot — Pexels"

[2]: https://www.pexels.com/license/ "Pexels license"

[3]: https://www.youtube.com/watch?v=3y82dHb9kVI "Parallel Parking — The Reverse Camera Technique"

[4]: https://sites.google.com/berkeley.edu/dlp-dataset "Dragon Lake Parking (DLP) Dataset"

[5]: https://github.com/KejiaGao/e2e-parking-carla-dataset "E2E Parking Dataset GitHub repository"

[6]: https://arxiv.org/html/2504.10812v2 "E2E Parking Dataset: An Open Benchmark for End-to-End Autonomous Parking"
