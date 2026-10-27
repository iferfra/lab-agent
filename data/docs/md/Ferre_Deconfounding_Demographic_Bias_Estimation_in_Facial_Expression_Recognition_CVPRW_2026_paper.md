This CVPR Workshop paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

# **Deconfounding demographic bias estimation in facial expression recognition** 

Iv´an Ferre<sup>1</sup> , Roberto Valle<sup>1</sup> , Jose´ M. Buenaposada<sup>2</sup> and Luis Baumela<sup>1</sup> 

> 1Departamento de Inteligencia Artificial, Universidad Polit´ecnica de Madrid, Spain 

- 2Departamento de Inform´atica y Estad´ıstica, Universidad Rey Juan Carlos, Spain https://pcr-upm.github.io 

josemiguel.buenaposada@urjc.es 

## **Abstract** 

_Ensuring demographic fairness has become a critical requirement for face processing technologies, and particularly for Facial Expression Recognition (FER). While it is well-established in the literature that visual confounders like head pose strongly affect the recognition performance of FER algorithms, their impact on the estimation of fairness has been overlooked. If demographic groups are observed under unequal visual conditions, usual performance comparisons can be heavily confounded. In this work, we analyze the impact of prominent visual confounders such as head pose and illumination on the estimation of gender bias. Through a controlled study on the Multi-PIE dataset, we show that unequal distributions of just one confounding factor can significantly distort the estimation of demographic bias. To address this, we propose a simple confounder-standardized evaluation protocol and test it on three in-the-wild datasets: Aff-Wild2, RAF-DB and AffectNet+. Our results reveal that the confounding effect is dataset-dependent. For example, in Aff-Wild2, the empirical fairness gap under head pose confounding increases from 4.23% to 6.25% once standardized. Ultimately, our findings demonstrate that fairness conclusions in FER are incomplete without accounting for confounding factors related to visual acquisition conditions, and we provide a practical framework for a more reliable bias assessment._ 

## **1. Introduction** 

Fairness is becoming an increasingly important topic in Facial Expression Recognition (FER), particularly for systems deployed in real-world scenarios and used across diverse demographic groups. It is usually measured by directly comparing aggregated performance metrics between groups, for example, by reporting recall or accuracy gaps between men and women [19, 29]. Differences are usually interpreted as evidence of bias. However, this comparison ignores an important issue: demographic groups may not be evaluated under the same visual conditions. 











<!-- Start of picture text -->
Pose 1 Pose 4 Pose 9 Pose 5<br>Illum. 7 Illum. 7 Illum. 6 Illum. 7<br>Neutral Smile Neutral Smile<br>Male Male Female Female<br>White White White White<br>(a) Multi-PIE [7]<br>[-27 ◦ , -12 ◦ , 0 ◦ ] [0 ◦ , 22 ◦ , 0 ◦ ] [25 ◦ , -4 ◦ , -2 ◦ ] [-45 ◦ , -7 ◦ , -9 ◦ ]<br>118 97 150 148<br>Neutral Neutral Other Surprise<br>Female Male Male Male<br>White White Black White<br>(b) Aff-Wild2 [13–16]<br>[-7 ◦ , -3 ◦ , 1 ◦ ] [-10 ◦ , -19 ◦ , 2 ◦ ] [-10 ◦ , -20 ◦ , 0 ◦ ] [0 ◦ , -17 ◦ , -1 ◦ ]<br>108 66 168 163<br>Sad Happy Happy Surprise<br>Female Male Female Female<br>White Black Asian White<br>(c) RAF-DB [17]<br>[6 ◦ , -2 ◦ , -4 ◦ ] [7 ◦ , -9 ◦ , -2 ◦ ] [-1 ◦ , -3 ◦ , -1 ◦ ] [-15 ◦ , -8 ◦ , 5 ◦ ]<br>149 63 137 81<br>Fear Surprise Contempt Happy<br>Male Male Female Female<br>White Black White Black<br>(d) AffectNet+ [6]<br><!-- End of picture text -->

Figure 1. Representative faces from Multi-PIE [7], Aff-Wild2 [13– 16], RAF-DB [17] and AffectNet+ [6] datasets illustrating variations in facial expressions, demographic groups, and visual confounders. Blue labels denote the visual confounders (head pose and illumination), and black labels the demographic attributes. 

5257 

Real-world FER data contains large variations in _visual confounding factors_ , such as head pose, illumination, and overall image quality related to the effect of occlusions, noise, and blur, as seen in Fig. 1. These factors directly affect the visibility of facial cues and consequently the difficulty of the recognition task. As a result, a fairness gap may reflect not only differences in model behavior across demographic groups but also differences in the visual conditions under which those groups are observed. Accordingly, usual fairness evaluation can be heavily confounded by visual factors. 

The magnitude of the impact of visual confounders depends on two conditions: first, FER performance must vary across the visual factor being analyzed; second, the factor distributions among the demographic subgroups must differ. When subgroup distributions are similar or FER is not affected by the visual factor, it is expected that the empirical and true fairness estimates will remain close. In contrast, when subgroup distributions are heavily imbalanced and the visual factor impacts performance, the empirical fairness can deviate strongly from the true fairness. This is a crucial point for interpreting dataset-specific bias. 

In this work, we study visual factors as potential confounders in the evaluation of gender fairness for FER. Specifically, we analyze the impact of head pose and illumination that have been recently identified as being among the factors that most severely affect algorithmic performance [21]. Rather than treating visual factors merely as sources of overall performance degradation, our goal is to draw attention to how and when unequal visual acquisition conditions can skew the estimation of demographic bias. To account for this, we introduce a confounder-standardized protocol that first measures performance within each factor subset and then recomputes subgroup performance under a balanced distribution. This effectively deconfounds the metrics, making group comparisons more consistent and providing a clearer estimate of intrinsic disparity. 

We validate this approach in both controlled and in-thewild settings. In Multi-PIE, we design a controlled experiment to show that fairness gaps can change significantly without modifying the model or its predictions, solely due to differences in visual condition distributions. We then analyze three in-the-wild FER datasets: Aff-Wild2, RAF-DB and AffectNet+. Our results show that the impact of visual confounders is dataset-dependent. It is more pronounced in Aff-Wild2, especially for head pose, and less marked in AffectNet+ and RAF-DB, where visual factor distributions per demographic subgroup are naturally more aligned. 

Overall, our results show that demographic bias estimation in FER can be skewed by unequal visual conditions, although the magnitude of this effect varies across datasets and confounding factors. Our standardized evaluation provides a simple, model-agnostic way to disentangle intrinsic 

model bias from dataset composition, leading to more reliable fairness measurements. 

Our main contributions are as follows: 

- We identify visual factors, such as head pose and illumination, as critical confounders in fairness evaluation for FER. 

- We propose a simple, confounder-standardized evaluation protocol to deconfound bias estimation and compare demographic groups under shared visual conditions. 

- Through a controlled experiment on Multi-PIE, we demonstrate that an unequal distribution of head poses in the test set alone can artificially inflate the measured fairness gap. 

- We analyze three in-the-wild FER datasets, demonstrating that the confounding effect is dataset-dependent and can mask underlying demographic disparities when subgroup factor distributions differ significantly. 

## **2. Related Work** 

This work lies at the intersection of (i) fairness evaluation in FER, (ii) robustness to visual factors, and (iii) bias induced by subgroup imbalance in face analysis datasets. We briefly review these three areas below and highlight how our work differs from previous studies. 

### **2.1. Fairness in FER** 

Traditionally, dataset bias has been widely studied when training and evaluating across different datasets [24]. Common solutions include combining datasets or using synthetic data to improve generalization [11, 25]. In this work, we instead focus on _demographic bias_ , which has become essential in face processing technologies, particularly in facial expression recognition (FER), as evidenced by the increasing attention from the research community [2, 4, 5, 10, 19, 29]. This line of research has identified that bias does not have a single origin but arises from a combination of factors, including dataset composition, representational bias, stereotypical bias, and annotation-related effects. A fundamental challenge in FER fairness is the subjective nature of the ground-truth labels themselves. Chen _et al_ . [2] demonstrated that human subjectivity significantly contaminates the labeling process ( _e.g_ ., more likely to perceive female faces as happier than male faces). These biases impact how models learn to generalize across demographic groups. 

Beyond annotation, the literature has largely focused on identifying performance disparities across demographic groups, with a particular emphasis on gender. Studies by Xu _et al_ . [29] and Manresa-Yee _et al_ . [19] have shown that standard FER architectures consistently fail to achieve equal accuracy across genders, often favoring the majority group in the training data. The usual approach in [19, 29] to measuring fairness consists of directly comparing aggregated performance metrics ( _e.g_ ., accuracy or recall gaps) between 

5258 

demographic groups. Although effective in surfacing disparities, this paradigm rests on a critical assumption: that groups are being compared under equivalent visual conditions. Our work builds upon these foundational studies but argues that without accounting for visual confounders ( _e.g_ ., head pose or illumination), these measured gaps may provide a distorted view of the model’s intrinsic bias. 

### **2.2. Visual Confounders in FER** 

The performance of face analysis models is known to be affected by the input image quality [20]. Furthermore, recent systematic audits of other foundational facial processing tasks, such as facial landmark detection, have confirmed that head pose and illumination are among the most dominant confounding factors driving performance disparities [21]. In particular, FER performance is known to be sensitive to environmental and geometric factors [23, 26], referred to as visual confounders, being the most documented head pose. Variations in yaw and pitch lead to selfocclusion and appearance variations that hinder expression decoding [26, 27]. While the impact of these factors on overall performance is well-known, their role in bias estimation remains overlooked. If demographic groups in a dataset do not have identical head pose and illumination distributions, a difference in performance between them may be caused by the different confounder distributions rather than by intrinsic demographic bias. As noted by Wu _et al_ . [28], a “balanced” dataset requires uniformity not only in demographic labels but also in the difficulty of the samples. Our work extends this logic by proposing that a reliable fairness evaluation requires comparing groups under standardized visual conditions. 

### **2.3. Data Composition and Subgroup Bias** 

When demographic subgroups are unevenly distributed across challenging ‘in-the-wild’ conditions ( _e.g_ ., extreme yaw or low illumination), the resulting performance gap is not a pure measure of demographic bias, but a reflection of subgroup imbalance in visual factors. In face analysis, previous work [1, 28] has highlighted that balancing a dataset only at the demographic level may not be sufficient if other important variables remain unbalanced. This perspective is especially relevant for fairness evaluation, since an empirical demographic gap may reflect both intrinsic model behavior and differences in the visual conditions under which each subgroup is evaluated. 

While previous studies focus on identifying performance gaps [19, 29] or proposing algorithmic mitigation [3, 22], our work focuses on the evaluation process itself. Specifically, we address a critical flaw in demographic bias estimation: the compositional confounding that occurs when visual factors and demographic labels are correlated. By introducing a standardized protocol, we isolate these effects, 

ensuring that the estimated bias reflects the model’s intrinsic behavior rather than an artifact of skewed data composition. 

## **3. Standardized Fairness Evaluation** 

We study fairness from an equal-opportunity perspective [8] by comparing class-wise true positive rates across demographic subgroups, while explicitly accounting for the distribution of visual confounders ( _e.g_ ., head pose, illumination). 

Let _C_ denote the set of facial expression classes, _G_ the demographic groups, and _B_ the set of bins of the confounder variables. We use confusion matrices as the basic concept from which all performance measures are derived. 

For each subgroup _g ∈G_ and visual-confounder bin _b ∈ B_ , we define the conditioned confusion matrix 



where each entry _Ci,j_<sup>_g,b_denotesthenumberofsamples</sup> whose ground-truth class is _i_ and whose predicted class is _j_ , where ( _i, j_ ) _∈C_ , restricting the evaluation to samples in subgroup _g_ and visual-confounder bin _b_ . 

From this matrix, the class-wise true positive rate (TPR) under condition ( _g, b_ ) is defined as 



This quantity corresponds to the recall of class _c_ within subgroup _g_ and visual condition _b_ . It captures how recognition performance varies across visual conditions for each subgroup. 

The usual evaluation protocol ignores the visual confounding variables, defining 



Our key insight is that a demographic subgroup _g_ may be exposed to different distributions of visual confounders _b_ and hence the _empirical macro-averaged_ TPR 



and the _empirical fairness gap_ for the gender demographic attribute, treated as a binary category _G_ = _{m, f }_ , 



may be confounded. 

5259 

To isolate model bias from these compositional effects, we introduce a standardized evaluation in which every subgroup is assessed under a balanced distribution of confounder bins. To summarize performance across expression classes, we compute the subgroup macro-averaged TPR as 



the _standardized macro-averaged_ TPR as 



and the _standardized fairness gap_ as 



where positive values indicate better male performance and negative values better female performance. 

This standardized estimate evaluates all groups under an identical distribution of visual confounders and therefore isolates performance differences that cannot be attributed to subgroup-specific visual composition. 

Finally, we quantify the effect of visual-confounder composition as 



If ∆ _≈_ 0, the estimated subgroup disparity is stable under variations of the visual confounders. If ∆ _>_ 0, the distribution of confounding factors inflates the estimated bias. However, if ∆ _<_ 0, the distribution of confounding factors masks part of the bias. 

## **4. Experimental Setup** 

We evaluate the proposed factor-standardized fairness protocol in both controlled and in-the-wild settings. For the controlled study, we use Multi-PIE [7], which provides a suitable benchmark to analyze the effect of head pose under structured acquisition conditions. This controlled environment allows us to isolate the impact of head pose by artificially varying its distribution across demographic groups in the test set. By doing so, we can demonstrate how the measured fairness gap changes due to dataset composition. 

For the in-the-wild analysis, we use Aff-Wild2 [13–16], AffectNet+ [6], and RAF-DB [17]. These datasets exhibit different degrees of variability in head pose and illumination, making them suitable for studying whether visual confounders can influence fairness evaluation in realistic conditions. 

Expression label spaces vary across datasets. Multi-PIE includes six categories (neutral, smile, surprise, squint, disgust, scream), RAF-DB defines seven (neutral, happy, surprise, fear, disgust, anger, sad), AffectNet+ comprises eight 

(neutral, happy, surprise, fear, disgust, anger, sad, contempt), and Aff-Wild2 uses a similar set (neutral, happy, surprise, fear, disgust, anger, sad, other), replacing ‘contempt’ with ‘other’. Because of these differences, our analysis is performed independently on each dataset, and fairness comparisons are made within each benchmark rather than across datasets at the class level. 

For each dataset, we perform subgroup analysis by gender using the available expression annotations. The same fairness evaluation protocol is applied across all datasets to ensure that the comparison is consistent. 

### **4.1. Demographic and Visual Factor Annotation** 

Not all datasets used in this work provide explicit demographic annotations. For datasets where the gender label is not available, as in Aff-Wild2, we obtain it automatically using FairFace [12]. This label is used only for the subgroup-based fairness evaluation and not as a training target for the FER models. 

Head pose annotations are not available for all datasets, so we estimate them from facial landmarks extracted with MediaPipe Face Mesh [18]. Specifically, we compute head orientation from three stable facial points ( _i.e_ ., left eye, right eye, and chin). In this work, we restrict the head pose representation to yaw, as it best captures the left-right rotation that influences the visibility of facial expression cues. 

In addition to head pose, we analyze illumination following the procedure in [28]. First, the face region is segmented using _BiSeNet_ [30]. Illumination is then approximated by computing a coarse brightness estimate over the segmented face region. To ensure a reliable estimate, images with a predicted face area less than 20% of the image or without a valid nose segmentation are excluded. 

These estimates provide a simple and consistent set of visual factors across datasets. Since our goal is to compare fairness under shared visual conditions rather than to recover precise physical measurements, these approximations are sufficient for the proposed factor-standardized evaluation. Code and dataset labels are available at https:// github.com/pcr-upm/cvprw26_recognition. 

### **4.2. Model and Training Protocol** 

To evaluate fairness under a consistent experimental setup we use the same FER architecture and training strategy across datasets. In line with Xu _et al_ ., [29], we train a ResNet-50 architecture [9] from scratch to avoid biases that may be inherited from large-scale pretraining data. 

Input images are resized to 100 _×_ 100, and models are trained with a batch size of 256 for up to 100 epochs. We use Adam with an initial learning rate of 10<sup>_−_3</sup> . 

During training, we apply geometric and photometric data augmentations, including random cropping to 96% of the image, random rotations within _±_ 15<sup>_◦_</sup> , horizontal flip- 

5260 

ping with 50% probability, and histogram equalization with 50% probability. Images are subsequently converted to tensors and normalized. 

The same trained model is used for both the empirical and the standardized fairness analyses, so that any difference between both estimates is due to the evaluation protocol and not to a change in the model. 

### **4.3. Visual Confounding Factor Binning** 

To obtain stable subgroup estimates, we discretize each visual confounding factor into a finite number of bins. 

For in-the-wild datasets, we divide head pose into two absolute-yaw bins: [0<sup>_◦_</sup> _,_ 15<sup>_◦_</sup> ) for near-frontal faces and [15<sup>_◦_</sup> _,_ 90<sup>_◦_</sup> ] for faces with larger yaw. For Multi-PIE, we similarly use two head pose bins: one for frontal and one for profile-like images. 

Regarding illumination, we use two average-intensity bins, [0 _,_ 125) and [125 _,_ 255], to distinguish between less and more illuminated faces. 

### **4.4. Evaluation Protocol** 

We evaluate fairness from an equal-opportunity perspective using the methodology defined in Section 3. For each demographic subgroup, we first compute the empirical macroaveraged TPR (3) from the class-wise subgroup recalls (2). 

We then compute a standardized macro-averaged TPR for each demographic subgroup by first evaluating performance within each confounder bin (5) and then aggregating across bins (6). 

In the binary gender setting, fairness is summarized through the signed difference between demographic subgroup macro-TPR values. For each confounding factor, we report the empirical fairness gap (4), the standardized fairness gap (7), and their difference (8), which quantifies the compositional effect induced by the distribution of the visual confounder in the dataset. 

## **5. Controlled Study on Multi-PIE** 

We first evaluate the proposed protocol in a controlled setting using Multi-PIE, where head pose can be explicitly controlled. We construct both training and test sets balanced across expressions, gender and head pose. 

To simulate different evaluation conditions, we construct evaluation subsets with controlled gender-specific head pose mixtures from the same balanced test pool. In the _balanced_ subset, both genders follow a 50% frontal / 50% profile distribution. In the _biased_ subset, head pose is correlated with gender, with males following an 80% frontal / 20% profile distribution and females a 20% frontal / 80% profile distribution. This design allows us to isolate the compositional effect of head pose on the measured fairness gap. 

|**Gender**|**Head**|**Pose**|
|---|---|---|
||**Frontal**|**Profile**|
|Female|69.01|68.72|
|Male|76.60|68.51|
|**GAP**|**7.59**|**-0.21**|



Table 1. Macro-averaged TPR (%) per gender on Multi-PIE using the head pose as visual confounder. 

|**Database**|**GAP**|**GAP**<sup>_std_</sup>|∆|
|---|---|---|---|
|Multi-PIE (balanced)|4.51|4.60|-0.09|
|Multi-PIE (biased)|6.39|4.29|2.10|



Table 2. Multi-PIE balanced and biased comparison of empirical and standardized fairness gaps (%) for gender under head pose confounding factor. 

### **5.1. Pose-Dependent Performance** 

Before comparing fairness under different head pose distributions, we first analyze FER performance separately for each gender and head pose. Results in Table 1 show that profile views are harder than frontal views for both groups, leading to lower macro-TPR in profile pose. More importantly, the gender gap is not constant in pose conditions. In frontal pose, the difference between male and female performance is about 7.59%, whereas in profile pose it is only around -0.21%. Therefore, the final fairness gap necessarily depends on the proportion of frontal and profile samples in the test set. 

This observation establishes the mechanism behind head pose confounding. Because the overall recognition difficulty varies significantly across poses, with frontal views being easier than profiles, any imbalance in how these poses are distributed across demographic groups will artificially skew the aggregated metric. Furthermore, since the subgroup gap itself also fluctuates across poses, the observed empirical fairness becomes highly sensitive to the dataset’s specific pose composition. 

### **5.2. Empirical vs. Pose-Standardized Fairness** 

We now compare the fairness gaps obtained under the two evaluation subsets. As shown in Table 2, the balanced subset yields an empirical fairness gap of 4.51% and a standardized gap of 4.60%, showing close agreement when both genders are evaluated under the same 50% frontal / 50% profile mixture. 

In contrast, in the biased subset the empirical fairness gap increases to 6.39%. This follows directly from the posedependent results in Table 1: frontal views are easier overall, and the male advantage is much larger in frontal pose 

5261 

|**Database**|**Gender**|**Head**|**Pose**|**Illum**|**ination**|
|---|---|---|---|---|---|
|||**0º-15º**|**+15º**|**0-125**|**125-255**|
|AffWild2|Female|32.10|22.29|27.86|30.57|
|-|Male|34.13|32.75|32.10|33.74|
||**GAP**|**2.03**|**10.46**|**4.24**|**3.17**|
|AffectNet+|Female<br>Male|48.26<br>50.61|43.78<br>47.63|47.67<br>48.78|47.49<br>51.30|
||**GAP**|**2.35**|**3.85**|**1.11**|**3.81**|
|RAFDB|Female|73.05|72.46|75.54|68.97|
|-|Male|72.26|69.77|72.41|71.92|
||**GAP**|**-0.79**|**-2.69**|**-3.13**|**2.95**|



Table 3. Macro-averaged TPR (%) per gender for each in-the-wild dataset using the head pose and illumination as visual confounders. 

than in profile pose. Therefore, when the evaluation subset contains a higher proportion of frontal male samples and profile female samples, the empirical fairness gap grows. However, the standardized fairness gap in the biased subset is 4.29%, which remains much closer to the empirical fairness gap observed under the balanced condition. 

Since both evaluation conditions are drawn from the same balanced test pool and use the same model, these differences can only be attributed to the head pose composition of the evaluation subset. Overall, this controlled experiment shows that head pose acts as a visual confounder in fairness evaluation. More importantly, it shows that when the evaluation data cannot be fully balanced, the proposed standardized metric can recover a fairness estimate that remains close to the one obtained under balanced conditions, and therefore better reflects the underlying subgroup disparity. 

## **6. In-the-Wild Analysis** 

We next evaluate the proposed standardized protocol using in-the-wild FER datasets, where visual conditions are not controlled and subgroup composition is determined by the dataset itself. In this setting, the measured fairness gap may depend not only on subgroup-specific FER performance, but also on how visual confounders such as head pose and illumination are distributed across demographic subgroups. 

### **6.1. Aff-Wild2** 

Aff-Wild2 provides a particularly informative in-the-wild case study, as head pose remains a relevant source of variability across the dataset. As shown in Fig. 2, head pose differs noticeably across gender: female samples are more concentrated in frontal positions, whereas male samples appear more frequently in profile views. In contrast, illumination remains more consistent across genders. 

Head pose also shows a clearer confounding pattern in 

|**Database**|**Visual factor**|**GAP**|**GAP**<sup>_std_</sup>|∆|
|---|---|---|---|---|
|Aff-Wild2|Head pose<br>Illumination|4.23|6.25<br>3.71|-2.02<br>0.52|
|AffectNet+|Head pose<br>Illumination|2.82|3.10<br>2.46|-0.28<br>0.36|
|RAF-DB|Head pose<br>Illumination|-0.82|-1.74<br>-0.09|0.92<br>-0.73|



Table 4. Dataset-level comparison of empirical and standardized fairness gaps (%) for gender under head pose and illumination confounding factors. 



<!-- Start of picture text -->
1.0<br>Pose bin<br>0–15°<br>0.8 +15°<br>0.6<br>0.4<br>0.2<br>0.0<br>Females Males<br>Gender<br>1.0<br>Illumination bin<br>0–125<br>0.8 125–255<br>0.6<br>0.4<br>0.2<br>0.0<br>Females Males<br>Gender<br>Proportion<br>Proportion<br><!-- End of picture text -->

Figure 2. Distributions of confounding factors in the Aff-Wild2 test set across gender (Females, Males) demographic subgroups. Top: head pose distributions. Bottom: illumination distributions. 

Aff-Wild2. As reported in Table 3, the Macro-averaged TPR gap between genders is substantially larger in the +15<sup>_◦_</sup> pose bin than in the 0<sup>_◦_</sup> to 15<sup>_◦_</sup> bin, indicating that the disparity is not uniform across pose conditions. Combined with the gender differences in pose distribution shown in Fig. 2, this suggests that head pose acts as a stronger confounding factor in this dataset. 

This is reflected in Table 4, where the confounding effect of head pose is _−_ 2 _._ 02%, with the fairness gap increasing 

5262 

from an empirical value of 4 _._ 23% to a standardized value of 6 _._ 25%. This negative value indicates that the empirical pose distribution was partially masking pose-induced bias. In particular, females perform better under frontal poses but worse under profile views, and the dataset is dominated by frontal samples. Consequently, averaging under the empirical pose distribution tends to favor females, whereas standardization reveals a larger underlying disparity. By contrast, male performance remains relatively stable across pose conditions. Overall, the original head pose distribution was reducing the empirical fairness gap. Regarding illumination, Table 3 shows some variation in the performance gap across illumination bins, although this variation is smaller than that observed for head pose. At the same time, Fig. 2 indicates that illumination is distributed very similarly across genders. Taken together, these results suggest that illumination has a weaker confounding effect than head pose in Aff-Wild2. 

Table 4 shows that the empirical fairness gap is 4 _._ 23%, while the standardized fairness gap is 3 _._ 71%, corresponding to an illumination confounding effect of 0 _._ 52%. Although smaller than the effect of head pose, this result indicates that illumination still has a measurable influence on the fairness metric, reinforcing the importance of accounting for confounding visual factors. 



<!-- Start of picture text -->
1.0<br>Pose bin<br>0–15°<br>0.8 +15°<br>0.6<br>0.4<br>0.2<br>0.0<br>Females Males<br>Gender<br>1.0<br>Illumination bin<br>0–125<br>0.8 125–255<br>0.6<br>0.4<br>0.2<br>0.0<br>Females Males<br>Gender<br>Proportion<br>Proportion<br><!-- End of picture text -->

Figure 3. Distributions of confounding factors in the AffectNet+ evaluation set across demographic subgroups. Top: head pose distributions by gender. Bottom: illumination distributions by gender. 

### **6.2. AffectNet+** 

AffectNet+ exhibits a smaller confounding effect than AffWild2, primarily because the majority of samples are concentrated in near-frontal poses (below 15º), with fewer than 20% of samples exceeding 15º. Regarding illumination, most images fall within the highest illumination bin, with only a slight difference between genders: male samples are somewhat more represented in the lower illumination bin, as shown in Fig. 3. 

In terms of head pose, Table 3 reports the performance gaps across pose bins for each gender. The gap is 2 _._ 35% in the 0<sup>_◦_</sup> –15<sup>_◦_</sup> range and 3 _._ 85% for poses above 15<sup>_◦_</sup> . However, since the head pose distribution is very similar across genders, the resulting confounding effect is limited. As shown in Table 4, the fairness gap increases only slightly from 2 _._ 82% to 3 _._ 10%, yielding a head pose confounding effect of _−_ 0 _._ 28%. 

For illumination, Table 3 reports a gap of 1 _._ 11% in the 0–125 bin and 3 _._ 81% in the 125–255 bin. Despite this difference, the overall illumination confounding effect remains small, at 0 _._ 36%. This is partly because female performance stays relatively similar across illumination bins, while male samples are more evenly distributed across illumination conditions than female samples. Even when the correction is modest, the proposed protocol is still valuable, as it reveals whether the reported fairness gap is robust to changes in confounding-factor composition. 

### **6.3. RAF-DB** 

RAF-DB presents a case similar to AffectNet+. Once again, the dataset is dominated by frontal poses, with fewer than 20% of samples exceeding 15<sup>_◦_</sup> , and the head pose distribution is highly similar across male and female groups. A similar pattern is observed for illumination: most samples fall within the higher illumination bin, although unlike AffectNet+, female samples are more represented in the lowerillumination bin than male samples. 

Table 3 shows a head pose gap of _−_ 0 _._ 79% in the first bin and _−_ 2 _._ 69% in the second bin. However, as in the previous dataset, head pose is distributed very similarly across genders and remains largely concentrated in frontal views, resulting in a limited overall confounding effect. Accordingly, Table 4 reports a head pose confounding effect of 0 _._ 92%, slightly larger than in AffectNet+. This can be explained by the more pronounced drop in male performance in the second pose bin, while female performance remains relatively stable across pose conditions. Consequently, averaging across pose bins amplifies the disparity. Since the fairness gap is negative in this dataset, meaning that females outperform males, the effect increases its magnitude, shifting the gap from _−_ 0 _._ 82% to _−_ 1 _._ 74%. 

Regarding illumination, Table 3 reveals a substantial change in the fairness gap across illumination bins. While females outperform males in the lower-illumination bin, it 

5263 



<!-- Start of picture text -->
1.0<br>Pose bin<br>0–15°<br>0.8 +15°<br>0.6<br>0.4<br>0.2<br>0.0<br>Females Males<br>Gender<br>1.0<br>Illumination bin<br>0–125<br>0.8 125–255<br>0.6<br>0.4<br>0.2<br>0.0<br>Females Males<br>Gender<br>Proportion<br>Proportion<br><!-- End of picture text -->

Figure 4. Distributions of confounding factors in the RAF-DB evaluation set across demographic subgroups. Top: head pose distributions by gender. Bottom: illumination distributions by gender. 

reverses in the brighter bin, where male performance becomes higher. As a result, the gap shifts from _−_ 3 _._ 13% to 2 _._ 95%, leading to an illumination confounding effect of _−_ 0 _._ 73%. Accordingly, the overall fairness gap moves from _−_ 0 _._ 82% to _−_ 0 _._ 09%. 

## **7. Conclusion** 

The main message of our work is that fairness in FER should not be interpreted independently of the visual conditions under which each demographic subgroup is evaluated. An empirical demographic gap may reflect intrinsic differences in model behavior, but it may also reflect differences in the visual conditions under which these groups are observed. Therefore, part of what is measured as demographic bias may actually be driven by differences in evaluation conditions rather than by subgroup behavior alone. 

The impact of visual confounding factors is strongly dataset-dependent. It becomes relevant when subgroup performance varies across the visual confounder being analyzed and when subgroup factor distributions differ. This explains why this effect is most prominent in the Multi-PIE synthetic experiment and in Aff-Wild2, a dataset showing diverse and unconstrained settings, while it is less evident in AffectNet+ and RAF-DB, where visual factor distributions are more aligned. Importantly, while some of these differences are small, controlled experiments demonstrate 

that fairness estimates can change significantly as a function of the confounders distributions. Further, although demographic annotations are obtained using an automated classifier and may contain noise, this does not affect the core finding, as our analysis isolates how dataset composition alone impacts the fairness metric while keeping the model’s predictions constant. 

These observations have implications for fairness evaluation and benchmark design. To assess fairness more reliably, evaluation datasets should include broader visual diversity and sufficient support across relevant factor ranges. When fully balanced evaluation sets are not available, the proposed factor-standardized protocol offers a simple and model-agnostic way to obtain fairness estimates that better reflect subgroup differences in model behavior rather than differences in visual evaluation conditions. 

Finally, while our analysis uses head pose, illumination, and gender as primary case studies, these findings apply equally to other visual confounders in FER, such as resolution and occlusion, as well as to other demographic variables, such as age and ethnicity. More broadly, the proposed confounder-standardized framework can be extended to virtually any face classification problem. Tasks such as face recognition, age estimation, and deepfake detection all rely on datasets where acquisition conditions frequently correlate with demographic attributes. By adopting confounderaware evaluation protocols, the broader face analysis community can ensure more accurate and interpretable demographic bias estimation. 

A primary limitation of our proposal is its heavy reliance on annotated confounding factors and sufficient sample density. Because our method requires evaluating performance within specific visual bins, increasing the number of confounders simultaneously ( _e.g_ ., pose, illumination, and resolution) exponentially increases the amount of data required to maintain stable estimates. In small or highly imbalanced datasets, this can lead to insufficient support for certain demographic-by-confounder combinations. Nevertheless, this extensive data requirement is an unavoidable necessity for rigorous fairness evaluation; without it, any measured bias remains inherently entangled with dataset composition. This highlights a critical need for the community to collect and annotate large-scale datasets with both demographic and environmental diversity in mind. 

## **Acknowledgements** 

This work was supported by project PID2022-137581OBI00 funded by MICIU/AEI/10.13039/501100011033 and FEDER, UE. L. Baumela and J. M. Buenaposada are members of the ELLIS Unit Madrid, funded by the Autonomous Community of Madrid. 

5264 

## **References** 

- [1] V´ıtor Albiero, Krishnapriya K. S, Kushal Vangara, Kai Zhang, Michael C. King, and Kevin W. Bowyer. Analysis of gender inequality in face recognition accuracy. In _Proc. Winter Conference on Applications of Computer Vision Workshops_ , pages 81–89, 2020. 3 

- [2] Yunliang Chen and Jungseock Joo. Understanding and mitigating annotation bias in facial expression recognition. In _Proc. International Conference on Computer Vision_ , pages 14960–14971, 2021. 2 

- [3] Jiaee Cheong, Sinan Kalkan, and Hatice Gunesn. Counterfactual fairness for facial expression recognition. In _Proc. European Conference on Computer Vision Workshops_ , pages 245–261, 2022. 3 

- [4] Iris Dominguez-Catena, Daniel Paternain, and Mikel Galar. Metrics for dataset demographic bias: A case study on facial expression recognition. _IEEE Trans. on Pattern Analysis and Machine Intelligence_ , 46(8):5209–5226, 2024. 2 

- [5] Iris Dominguez-Catena, Daniel Paternain, Aranzazu Jurio, and Mikel Galar. Less can be more: representational vs. stereotypical gender bias in facial expression recognition. _Prog. Artif. Intell._ , 14(1):11–31, 2025. 2 

- [6] Ali Pourramezan Fard, Mohammad Mehdi Hosseini, Timothy D. Sweeny, and Mohammad H. Mahoor. Affectnet+: A database for enhancing facial expression recognition with soft-labels. _IEEE Trans. on Affective Computing_ , 17(1):784– 800, 2026. 1, 4 

- [7] Ralph Gross, Iain Matthews, Jeffrey Cohn, Takeo Kanade, and Simon Baker. Multi-PIE. In _Proc. International Conference on Automatic Face and Gesture Recognition_ , pages 1–8, 2008. 1, 4 

- [8] Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. In _Proc. Annual Conference on Neural Information Processing Systems_ , pages 3315–3323, 2016. 3 

- [9] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ , pages 770–778, 2016. 4 

- [10] Mohammad Mehdi Hosseini, Ali Pourramezan Fard, and Mohammad H. Mahoor. Faces of fairness: Examining bias in facial expression recognition datasets and models. _CoRR_ , abs/2502.11049, 2025. 2 

- [11] Indu Joshi, Marcel Grimmer, Christian Rathgeb, Christoph Busch, Franc¸ois Br´emond, and Antitza Dantcheva. Synthetic data in human analysis: A survey. _IEEE Trans. on Pattern Analysis and Machine Intelligence_ , 46(7):4957–4976, 2024. 2 

- [12] Kimmo K¨arkk¨ainen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In _Proc. Winter Conference on Applications of Computer Vision_ , pages 1547–1557, 2021. 4 

- [13] Dimitrios Kollias and Stefanos Zafeiriou. Expression, affect, action unit recognition: Aff-wild2, multi-task learning and arcface. In _Proc. British Machine Vision Conference_ , page 297, 2019. 1, 4 

- [14] Dimitrios Kollias, Panagiotis Tzirakis, Mihalis A. Nicolaou, Athanasios Papaioannou, Guoying Zhao, Bj¨orn W. Schuller, Irene Kotsia, and Stefanos Zafeiriou. Deep affect prediction in-the-wild: Aff-wild database and challenge, deep architectures, and beyond. In _International Journal of Computer Vision_ , pages 907–929, 2019. 

- [15] Dimitrios Kollias, Panagiotis Tzirakis, Alan Cowen, Stefanos Zafeiriou, Irene Kotsia, Eric Granger, Marco Pedersoli, Simon Bacon, Alice Baird, Chris Gagne, Chunchang Shao, Guanyu Hu, Soufiane Belharbi, and Muhammad Haseeb Aslam. Advancements in affective and behavior analysis: The 8th ABAW workshop and competition. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition Workshops_ , pages 5572–5583, 2025. 

- [16] Dimitrios Kollias, Stefanos Zafeiriou, Irene Kotsia, Gregory G. Slabaugh, Damith Chamalke Senadeera, Jianian Zheng, Kaushal K. K. Yadav, Chunchang Shao, and Guanyu Hu. From emotions to violence: Multimodal fine-grained behavior analysis at the 9th ABAW. In _Proc. International Conference on Computer Vision Workshops_ , pages 1–12, 2025. 1, 4 

- [17] Shan Li and Weihong Deng. Reliable crowdsourcing and deep locality-preserving learning for unconstrained facial expression recognition. _IEEE Trans. on Image Processing_ , 28 (1):356–370, 2019. 1, 4 

- [18] Camillo Lugaresi, Jiuqiang Tang, Hadon Nash, Chris McClanahan, Esha Uboweja, Michael Hays, Fan Zhang, ChuoLing Chang, Ming Guang Yong, Juhyun Lee, Wan-Teh Chang, Wei Hua, Manfred Georg, and Matthias Grundmann. Mediapipe: A framework for building perception pipelines. _CoRR_ , abs/1906.08172, 2019. 4 

- [19] Cristina Manresa-Yee, Silvia Ramis-Guarinos, and Jose Maria Buades Rubio. Facial expression recognition: Impact of gender on fairness and expressions. In _Interacci´on 2022: XXII International Conference on Human Computer Interaction_ , pages 1–8, 2022. 1, 2, 3 

- [20] Nima Najafzadeh, Hossein Kashiani, Mohammad Saeed Ebrahimi Saadabadi, Niloufar Alipour Talemi, Sahar Rahimi Malakshan, and Nasser M. Nasrabadi. Face image quality vector assessment for biometrics applications. In _Proc. Winter Conference on Applications of Computer Vision Workshops_ , pages 511–520, 2023. 3 

- [21] Pablo Parte, Roberto Valle, Jos´e Miguel Buenaposada, and Luis Baumela. Auditing demographic bias in facial landmark detection for fair human-robot interaction. _CoRR_ , abs/2604.06961, 2026. 2, 3 

- [22] Syed Sameen Ahmad Rizvi, Aryan Seth, and Pratik Narang. Balancing the scales: Enhancing fairness in facial emotion recognition with latent alignment. In _Proc. International Conference on Pattern Recognition_ , pages 113–128, 2024. 3 

- [23] Giota Stratou, Abhijeet Ghosh, Paul E. Debevec, and LouisPhilippe Morency. Effect of illumination on automatic expression recognition: A novel 3d relightable facial database. In _Proc. International Conference on Automatic Face and Gesture Recognition_ , pages 611–618, 2011. 3 

- [24] Antonio Torralba and Alexei A. Efros. Unbiased look at 

5265 

dataset bias. In _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ , pages 1521–1528, 2011. 2 

- [25] Roberto Valle, Jos´e Miguel Buenaposada, and Luis Baumela. Reducing head pose estimation data set bias with synthetic data. _IEEE Access_ , 13:73530–73539, 2025. 2 

- [26] Shangfei Wang, Yi Wu, Yanan Chang, Guoming Li, and Meng Mao. Pose-aware facial expression recognition assisted by expression descriptions. _IEEE Trans. on Affective Computing_ , 15(1):241–253, 2024. 3 

- [27] Philipp Werner, Frerk Saxen, Ayoub Al-Hamadi, and Hui Yu. Generalizing to unseen head poses in facial expression recognition and action unit intensity estimation. In _Proc. International Conference on Automatic Face and Gesture Recognition_ , pages 1–8, 2019. 3 

- [28] Haiyu Wu and Kevin W. Bowyer. What should be balanced in a “balanced” face recognition dataset? In _Proc. British Machine Vision Conference_ , 2023. 3, 4 

- [29] Tian Xu, Jennifer White, Sinan Kalkan, and Hatice Gunes. Investigating bias and fairness in facial expression recognition. In _Proc. European Conference on Computer Vision Workshops_ , pages 506–523, 2020. 1, 2, 3, 4 

- [30] Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, and Nong Sang. Bisenet: Bilateral segmentation network for real-time semantic segmentation. In _Proc. European Conference on Computer Vision_ , pages 334–349, 2018. 4 

5266 

