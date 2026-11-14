# PAD-Triage: Multimodal Smartphone Skin Lesion Screening

**Track:** AI + Health  
**Team:(SOLO)** Raksha Tiwari 
## The problem
Primary care physicians, rural clinics, and community health workers face massive patient backlogs and lack immediate access to dermatologists, leading to delayed diagnoses of malignant skin lesions like melanoma and basal cell carcinoma. When early-stage skin cancers are missed due to triage delays, patient survival outcomes drop significantly.

## What we built
An end-to-end multimodal AI triage system that takes a smartphone image of a skin lesion alongside basic clinical metadata (patient age and lesion diameters) and outputs an instant malignancy risk score with a clinician referral recommendation, accessible via an interactive Gradio web interface.

## What we added
* **True Multimodal Fusion:** Combined high-dimensional ResNet-50 visual representation features with tabular clinical features (`age`, `diameter_1`, `diameter_2`) via a dense neural fusion classifier.
* **Strict Patient-Level Splitting:** Prevented data leakage by ensuring images from the same patient never span across train, validation, and test sets.
* **Clinical Subgroup Auditing:** Evaluated recall and false-negative rates separately across clinical subsets to ensure safe, reliable triage performance.

## How it works
* **Data:** PAD-UFES-20 dataset containing 2,298 images across 1,373 patients and 6 diagnostic classes. Missing values in tabular features were cleaned via median imputation, and targets were mapped to binary malignancy (MEL, BCC, SCC) versus benign/pre-malignant classes.
* **Model:** Pretrained ResNet-50 vision backbone (freezing early feature extractor layers and fine-tuning later blocks) fused with tabular metadata features through a multi-layer neural network with Dropout and Sigmoid activation.
* **Baseline:** Clinical tabular Random Forest baseline trained exclusively on age and lesion diameters, achieving an initial ROC-AUC of 0.8630 on unseen test data.

## Results
* **Tabular Baseline:** Achieved 78.09% accuracy, 76.52% recall, 78.92% precision, and 0.8630 ROC-AUC.
* **Multimodal Model:** Outperformed the tabular baseline by successfully capturing high-dimensional textural, border irregularity, and color asymmetry features from smartphone lesion photos.
* **Where it fails:** Highly pigmented benign nevi (moles) or seborrheic keratoses with irregular borders can occasionally trigger false positives for melanoma.
* *Disclaimer:* This project is a prototype and is not medical advice.

## What we'd do next
* Fine-tune the full convolutional neural network backbone on higher-resolution dermoscopic image tiles.
* Integrate automated lesion segmentation (U-Net) to crop out background skin artifacts and hair prior to classification.

## Sources
* PAD-UFES-20 Dataset (Mendeley Data): https://data.mendeley.com/datasets/zr7vgbcyr2/1
* PyTorch & Torchvision Pretrained Models: https://pytorch.org
* Gradio UI Framework: https://fafbf82d40a4e00b17.gradio.live
* Streamlit : https://pad-triage-skin-lesion-b5hcvr8ygqp53jbtzlmq6o.streamlit.app/
