# BioGuard-X Task 2 — Notebook كاملة + Python helpers + Flask game prompt

## أنت حملت 7 ZIPs بالفعل — ومش محتاج تنزل ملف Actor 08

المطلوب وجود الملفات الأصلية في **Downloads** بنفس أسمائها:

```
<DOWNLOADS>\Downloads\Video_Speech_Actor_01.zip
<DOWNLOADS>\Downloads\Video_Speech_Actor_02.zip
<DOWNLOADS>\Downloads\Video_Speech_Actor_03.zip
<DOWNLOADS>\Downloads\Video_Speech_Actor_04.zip
<DOWNLOADS>\Downloads\Video_Speech_Actor_05.zip
<DOWNLOADS>\Downloads\Video_Speech_Actor_06.zip
<DOWNLOADS>\Downloads\Video_Speech_Actor_07.zip
```

**سيبهم ZIP بدون فك ضغط**. الكود سيقرأ ملفات speech audio+video من الداخل ويستخرج WAV من نفس MP4 باستخدام FFmpeg المضمن في `imageio-ffmpeg`. يتجاهل نسخة الفيديو بدون صوت (modality 02) وملفات الأغاني، لذا يوفّر وقت ومساحة.

**الداتا الأصلية ومصدرها:** [RAVDESS — Zenodo](https://zenodo.org/records/1188976) — رخصة CC BY-NC-SA 4.0 (استخدام غير تجاري مع نسب المصدر). التدريب والتقييم هدفهما بحث/ديمو فقط.

## 1) فك ملف مشروع BioGuard الجديد في Downloads

لما تنزّل `BioGuard_Task2_COMPLETE.zip`، فكّه عشان يبقى المجلد:

```
<DOWNLOADS>\Downloads\BioGuard_Task2_COMPLETE\
    BioGuard_Full_Master_Notebook.ipynb
    setup_venv.ps1
    requirements.txt
    prepare_ravdess.py
    bioguard_core.py
    public_research.py
    deployment.py
    fingerprint_bonus.py
    commands_bonus.py
    live_verification.py
    CODEX_FLASK_GAME_PROMPT.md
    model\             (exports after training)
    outputs\           (charts after notebook runs)
    public_data\       (audio / image / video pairs, automatically prepared)
```

## 2) عمل Virtual Environment والـKernel — PowerShell

```powershell
cd "$env:USERPROFILE\Downloads\BioGuard_Task2_COMPLETE"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name bioguard-task2 --display-name "Python (BioGuard Task 2)"
.\.venv\Scripts\python.exe -m jupyter lab
```

بديل: `powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1` لتسجيل كل ده أوتوماتيك (بشرط Python 3.12 موجود مع `py` launcher). **مش لازم تعطل سياسة PowerShell على الجهاز كله**.

اختار الـKernel **Python (BioGuard Task 2)**، ثم افتح `BioGuard_Full_Master_Notebook.ipynb` واضغط **Run All**. مهم: أول تجهيز الفيديوهات قد يأخذ وقتًا حسب المعالج وسرعة القرص؛ ستظهر أرقام التقدم. أوقف التشغيل فقط لو في Error واضح.

لو الداتا في مكان غير Downloads:

```powershell
.\.venv\Scripts\python.exe prepare_ravdess.py --source "D:\RAVDESS-ZIPS"
```

وبعدها شغل النوتبوك كالعادة. `MAX_CLIPS_PER_ACTOR=28` داخل Notebook كبداية؛ يمكن ترفعه 40 لإعادة experiment أكبر، لكن **امسح `public_data/feature_cache.joblib` بعد تغيير إعدادات استخراج الـFeatures**.

## 3) الصدق العلمي والـSplits

- Actors 01..05 = Known identities. كل ممثل له enrollment reference clips منفصلة + train/val/test clips.
- Actor 06 = Unseen identity to calibrate UNKNOWN validation thresholds (not enrolled).
- Actor 07 = Unseen identity held out until FINAL TEST (not enrolled, not used for threshold choice).
- Positive trial = genuine actor matched to his reference; negative trial = wrong actor/template.
- تقسيم الفيديو والصوت والصورة قائم على **source clip/group** وليس الـframe، منعًا لتسريب نسخ نفس clip للـTest.
- Face + voice هما البيومتريك الأساسيين، Video = frames متعددة من الوجه، وليس biometric ثالث مستقل.
- نسب الـAccuracy/FAR/FRR لا توجد قبل استخراج البيانات وتشغيل التدريب على جهازك فعليًا.

## 4) Notebook تغطي كل المطلوب

EDA, preprocessing, HOG, LBP, MFCC+delta+delta2, ZCR, RMS, video frame pooling, SVM/RF for every modality/fusion, verification pairs, confusion matrices, ROC/AUC/EER, FAR/FRR, early image+voice, early image+video, early video+voice, early all 3, late fusion, sequential hierarchical, open-set unknown rejection, experiments, fingerprint bonus, speech commands bonus, live webcam+mic bonus, enrollment bonus, export.

**الـFingerprint:** RAVDESS ليس بها صور بصمات أصابع. يتم تشغيل كود Sobel+LBP فعليًا فقط عند توفير بيانات بصمات منفصلة حقيقية في `fingerprint_data/<user>/*.png`. لا ندمجها إحصائيًا مع أشخاص آخرين لا تتطابق هويتهم.

**Speech commands:** تتطلب صوت الأوامر المصنّف في `commands/<word>/train|val|test/*.wav`. عند عدم توفرها ستظهر `BONUS PENDING` وليس Accuracy وهمية. يمكن جمعها لاحقًا.

## 5) ملفات بتطلع بعد Run All

```
model/public_models.joblib               # verifier: trained model objects, templates, thresholds
model/model_manifest.json               # exact model names, modality mapping, threshold, feature design
model/public_model_comparison.csv        # ALL SVM vs RF experiments
model/public_mode_comparison.csv         # best-per-mode + late/hierarchical
model/open_set_comparison.csv            # UNKNOWN detection summary
model/open_set_test_predictions.csv      # detail for real held-out actors
outputs/01_dataset_exploration.png       # EDA etc through 11_ROC_EER.png
```

Notebooks/Flask share identical feature extractor code; **do not copy weights into random classifiers**. All `.joblib` files should be locally generated (pickle can execute code when loaded).

## 6) تسجيل موظف جديد بعد تشغيل الـNotebook

مجلد `model/enrollment_gallery.joblib` يبدأ **فاضي** تمامًا؛ ممثلو RAVDESS لا يدخلون اللعبة تلقائيًا.

من كود Flask الذي سينشئه Codex: Admin → Add User → 4 صور من كاميرا أو رفع + 4 تسجيلات صوت من مايك أو رفع → `deployment.enroll_user(name, face_paths, wav_paths, allowed=True)` → يضاف المستخدم وتُحدّث الـGallery فورًا **دون تدريب SVM من جديد**. مهم الصور والتسجيلات تكون للشخص نفسه بموافقته؛ وبعدها **امتحن على محاولات جديدة** منفصلة عن الأربع عينات.

التجربة في الطرفية (بعد ما تتسجل مستخدمين وتكون كاميرا/مايك متاحين):

```powershell
.\.venv\Scripts\python.exe live_verification.py
```

واجهة اللعبة الحقيقية هينفذها Codex باستخدام ملف التعليمات `CODEX_FLASK_GAME_PROMPT.md` المرفق، بعد انتهاء Notebook. استخدمه مع نفس فولدر المشروع بما فيه `model/public_models.joblib`.

## 7) فحص ومشاكل متوقعة

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

- `FileNotFoundError Video_Speech_Actor...`: راجع الملفات في Downloads واسمها وأن الأرشيف مكتمل.
- `Invalid ZIP`: التحميل مش كامل؛ اعمل Download للملف التالف فقط.
- `No face detected`: جرّب عدد عينات أكبر `MAX_CLIPS_PER_ACTOR=40` أو راجع صور المصدر والـOpenCV.
- `ModuleNotFoundError`: اختار Kernel الصحيحة: Python (BioGuard Task 2).
- `Video codec/ffmpeg`: مكتبة `imageio-ffmpeg` متضمنة في `requirements.txt`، ولا يلزم ffmpeg منفصل غالبًا.
- `PermissionError` في Downloads: انقل الـZIPs إلى `raw_ravdess/` وأعد التشغيل.
- `Early Fusion` train/test: لازم كل زوج صورة+صوت+فيديو يكون مطابق لنفس actor ونفس clip، والكود بيفرض ده.

**تنبيه:** ده نظام تعليمي، غير صالح وحده لحماية دخول مصنع حقيقي بسبب احتمالات خطأ الكشف، انتحال الصور/التسجيلات، انحياز الداتا، وتغير بيئة التسجيل. يحتاج liveness/hardware وsecurity assessment حقيقي قبل أي استخدام فعلي.
