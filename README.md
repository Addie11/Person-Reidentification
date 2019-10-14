# Person-Reidentification

![Test](./image/test.jpg)

“Given an image/video of a person taken from one camera, re-identification is the process of identifying the person from images/videos taken from a different camera with non-overlapping fields of views. Re-identification is indispensable in establishing consistent labeling across multiple cameras or even within the same camera to re-establish disconnected or lost tracks.” 

**Our Approach**
To tackle the stated problem, we used combination of two approaches:

**Person Re-identification using feature mapping:**
*	A resnet50 model is trained on openly available Market 1501 dataset
*	Trained on 12186 images
*	Validated on 751 images
*	Trained for 15 epochs
*	Pytorch framework used at the backend
*	We use the model to generate the feature map of each individual present on the image and store the feature map in the database after assigning the id.
*	This model returns a vector of size (1, 512)

**Person Re-identification by Attribute-Assisted Appearance**
*	Trained a resnet50 model on RAP v2.0 (Richly Annotated Dataset for Pedestrian Attribute Recognition) having 84929 well annotated images
*	Rap v2.0 is not freely available and we had to sign an agreement to use this dataset
*	The author of the paper https://www.researchgate.net/publication/301817457_A_Richly_Annotated_Dataset_for_Pedestrian_Attribute_Recognition, Dangwei Li helped us with the dataset
*	We trained our model to identify 82 attributes as follows:
[Female, AgeLess16, Age17-30, Age31-45, Age46-60, BodyFat, BodyNormal, BodyThin, Customer, Employee, hs-BaldHead, hs-LongHair, hs-BlackHair, hs-Hat, hs-Glasses, ub-Shirt, ub-Sweater, ub-Vest, ub-TShirt, ub-Cotton, ub-Jacket, ub-SuitUp, ub-Tight, ub-ShortSleeve, ub-Others, ub-ColorBlack, ub-ColorWhite, ub-ColorGray, up-ColorRed, ub-ColorGreen, ub-ColorBlue, ub-ColorSilver, ub-ColorYellow, ub-ColorBrown, ub-ColorPurple, ub-ColorPink, ub-ColorOrange, ub-ColorMixture, ub-ColorOther, lb-LongTrousers, lb-Skirt, lb-ShortSkirt, lb-Dress, lb-Jeans, lb-TightTrousers, lb-ColorBlack, lb-ColorWhite, lb-ColorGray, lb-ColorRed, lb-ColorGreen, lb-ColorBlue, lb-ColorSilver, lb-ColorYellow, lb-ColorBrown, lb-ColorPurple, lb-ColorPink, lb-ColorOrange, lb-ColorMixture, lb-ColorOther, shoes-Leather, shoes-Sports, shoes-Boots, shoes-Cloth, shoes-Casual, shoes-Other, attachment-Backpack, attachment-ShoulderBag, attachment-HandBag, attachment-Box, attachment-PlasticBag, attachment-PaperBag, attachment-HandTrunk, attachment-Other, action-Calling, action-Talking, action-Gathering, action-Holding, action-Pushing, action-Pulling, action-CarryingByArm, action-CarryingByHand, action-Other]
*	Trained a resnet50 model and got a MAP (Mean Average Precision of 71%)
*	Trained on 67,433 images
*	Validated on 16,589 images
*	Trained for 100 epochs
*	Model returns a vector of size (1, 82)

**Dataset**
* Market1501
* RAP V2.0 (Richly Annotated Dataset for Pedestrian Recognition

**Research Papers:**
*	http://jankautz.com/publications/JointReID_CVPR19.pdf
*	https://www.researchgate.net/publication/301817457_A_Richly_Annotated_Dataset_for_Pedestrian_Attribute_Recognition
*	https://paperswithcode.com/paper/alignedreid-surpassing-human-level
*	https://paperswithcode.com/paper/a-strong-baseline-and-batch-normalization

**Github Repositories:**
*	https://github.com/layumi/Person_reID_baseline_pytorch
*	https://github.com/dangweili/pedestrian-attribute-recognition-pytorch
*	https://github.com/wangxiao5791509/Pedestrian-Attribute-Recognition-Paper-List
*	https://github.com/hyk1996/Person-Attribute-Recognition-MarketDuke
*	https://github.com/yuange250/MARS-Attribute




