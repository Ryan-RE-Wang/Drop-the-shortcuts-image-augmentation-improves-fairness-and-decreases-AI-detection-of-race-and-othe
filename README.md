# Drop the shortcuts: image augmentation improves fairness and decreases AI detection of race and other demographics from medical images

## Paper
This work is published on eBiomedicine (https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(24)00082-3/fulltext.)

If you find this work useful, we would appreciate you citing this paper.
```
@article{wang2022early,
  title={Early Diagnosis of Chronic Obstructive Pulmonary Disease from Chest X-Rays using Transfer Learning and Fusion Strategies},
  author={Wang, Ryan and Chen, Li-Ching and Moukheiber, Lama and Moukheiber, Mira and Moukheiber, Dana and Zaiman, Zach and Moukheiber, Sulaiman and Litchman, Tess and Seastedt, Kenneth and Trivedi, Hari and others},
  journal={arXiv preprint arXiv:2211.06925},
  year={2022}
}
```

## Background

It has been shown that AI models can learn race on medical images, leading to algorithmic bias. Our aim in this study was to enhance the fairness of medical image models by eliminating bias related to race, age, and sex. We hypothesise models may be learning demographics via shortcut learning and combat this using image augmentation.

## Methods
This study included 44,953 patients who identified as Asian, Black, or White (mean age, 60.68 years ±18.21; 23,499 women) for a total of 194,359 chest X-rays (CXRs) from MIMIC-CXR database. The included CheXpert images comprised 45,095 patients (mean age 63.10 years ±18.14; 20,437 women) for a total of 134,300 CXRs were used for external validation. We also collected 1195 3D brain magnetic resonance imaging (MRI) data from the ADNI database, which included 273 participants with an average age of 76.97 years ±14.22, and 142 females. DL models were trained on either non-augmented or augmented images and assessed using disparity metrics. The features learned by the models were analysed using task transfer experiments and model visualisation techniques.
## Findings
In the detection of radiological findings, training a model using augmented CXR images was shown to reduce disparities in error rate among racial groups (−5.45%), age groups (−13.94%), and sex (−22.22%). For AD detection, the model trained with augmented MRI images was shown 53.11% and 31.01% reduction of disparities in error rate among age and sex groups, respectively. Image augmentation led to a reduction in the model's ability to identify demographic attributes and resulted in the model trained for clinical purposes incorporating fewer demographic features.
## Interpretation
The model trained using the augmented images was less likely to be influenced by demographic information in detecting image labels. These results demonstrate that the proposed augmentation scheme could enhance the fairness of interpretations by DL models when dealing with data from patients with different demographic backgrounds.
