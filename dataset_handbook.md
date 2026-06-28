# Universal Dataset & Sign Specification Handbook

**Target Compatibility:** Windows, Linux (Ubuntu), and macOS.

---

## 1. Directory Structure Rule (All Operating Systems)
To ensure scripts work seamlessly across Windows, Linux, and Mac without path errors, all directory names must be STRICTLY UPPERCASE with NO SPACES.

Project Directory Mapping:
dataset/
├── metadata/
└── raw_videos/
    ├── HELLO/
    ├── THANK_YOU/
    ├── HELP/
    ├── WATER/
    ├── FOOD/
    ├── YES/
    ├── NO/
    ├── PLEASE/
    ├── SORRY/
    └── STOP/

*Python Note:* Always use forward slashes ('/') in code paths.

---

## 2. Universal File Naming Convention
To prevent cross-OS casing errors, every recorded file must strictly follow this exact layout in ALL CAPS:

Format: [SIGN_NAME]_[MEMBER_ID]_[SAMPLE_NUMBER].mp4

Member ID Assignment:
- Akshita & Dheeraj (Windows): M1, M2
- Mohit & Armaan (Linux): M3, M4
- Hariom (macOS): M5

Examples:
- THANK_YOU_M1_001.mp4 (Windows)
- HELLO_M4_042.mp4     (Linux)
- STOP_M5_100.mp4      (Mac)

-----

## 3. Vocabulary Nature & Structural Protocol OR Kinetic Gesture
- HELLO     -> DYNAMIC (Moving hand gesture)
- THANK_YOU -> DYNAMIC (Moving hand gesture)
- HELP      -> STATIC (Freeze hand position)
- WATER     -> STATIC (Freeze hand position)
- FOOD      -> DYNAMIC (Moving hand gesture)
- YES       -> DYNAMIC (Moving hand gesture)
- NO        -> DYNAMIC (Moving hand gesture)
- PLEASE    -> DYNAMIC (Moving hand gesture)
- SORRY     -> DYNAMIC (Moving hand gesture)
- STOP      -> STATIC (Freeze hand position)

---

## 4. Quality Rules for Data Collection
1. **Lighting Profiles:** Every member must record across 3 lighting setups: Daylight, Indoor LED, and Low-Light.
2. **Backgrounds:** Change locations (minimum 3 different backgrounds) to avoid overfitting.
3. **Distance:** Maintain a steady distance of 1.0 to 2.0 meters from the webcam.