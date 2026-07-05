"""
This script converts a png(jpg) file in the specified directory to an eps file.
How to use
python3 png2eps.py <png_dir>
required Pillow
"""

from PIL import Image
from argparse import ArgumentParser
import glob
import os

def parser():
    parser = ArgumentParser(
        description="This script is convert png file to eps file")
    parser.add_argument("out_dir", help="Target output directory")
    parser.add_argument('--png_dir', help="Target image directory")
    parser.add_argument('--png', action="append", help="Target image")
    return parser

def parse_args():
    p = parser()
    args = p.parse_args()
    return args


def remove_transparency(im, bg_color=(255, 255, 255)):
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]

        bg = Image.new("RGBA", im.size, bg_color + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im

def collect_files(args):
    if args.png_dir is not None:
        img_list = glob.glob(args.png_dir+"/*.png")
        img_list += glob.glob(args.png_dir+"/*.jpg")
    if args.png is not None:
        img_list = []
        for png in args.png:
            img_list += glob.glob(png)

    return img_list

def convert(args):
    """
    Converts the .png file to .eps file
    """
    img_list = collect_files(args)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    for img in img_list:
        im = Image.open(img)
        if im.mode in ('RGBA', 'LA'):
            im = remove_transparency(im)
            im = im.convert('RGB')
        name = os.path.splitext(img)[0].split('\\')[-1]
        im.save(args.out_dir + '/' + name + ".eps", lossless=True)

def main_args(args):
    print("Converting png to eps...")
    convert(args)
    print("Conversion is finishing. \nOutput path is {}".format(args.out_dir))

def main():
    args = parse_args()
    main_args(args)



if __name__ == "__main__":
    p = parser()

    args = p.parse_args(["C:\\Users\\Daniel\\Downloads",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\denseLoss\\iterate\\alpha_0.90\\Scaled_Predictions_F1_0.7272727272727272.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\denseLoss_retrained_rRT\\alpha_0.70\\Scaled_Predictions_F1_0.8000000000000002_4.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\denseLoss_retrained_autoencoder_ss\\alpha_0.60\\Scaled_Predictions_F1_0.8000000000000002_4.png",


                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\using_training_richardson_alongside\\retrained_regNN_oversampled_0_6\\Scaled_Predictions_F1_0.4.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\using_training_richardson_alongside\\retrained_rRT_0_3\\Scaled_Predictions_F1_0.5000000000000001_2.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\using_training_richardson_alongside\\retrained_autoencoder_ss_0_2\\Scaled_Predictions_F1_0.5454545454545454_4.png",

                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\richardson_mixed\\retrained_regNN_oversampled_0_5\\Scaled_Predictions_F1_0.22222222222222224.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\richardson_mixed\\retrained_rRT_0_6\\Scaled_Predictions_F1_0.22222222222222224_1.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\richardson_mixed\\retrained_autoencoder_ss_0_5\\Scaled_Predictions_F1_0.6666666666666666.png",

                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\retrained_regNN_oversampled_0_7\\Scaled_Predictions_F1_0.6_1.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\retrained_rRT_0_1\\Scaled_Predictions_F1_0.7272727272727272.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\retrained_autoencoder_ss_0_1\\Scaled_Predictions_F1_0.7272727272727272_3.png",


                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\dl_rRegNN_F1_vs_alpha.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\dl_rRT_F1_vs_alpha.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\dl_rRT_AE_F1_vs_alpha.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\re_rRegNN_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\re_rRT_AE_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\re_rRT_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\rc_rRegNN_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\rc_rRT_AE_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\rc_rRT_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\rRegNN_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\rRT_AE_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\rRT_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\cRegNN_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\cRT_AE_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\cRT_F1_vs_Oversampling_Rate.png",



                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Predicted_Peak_Intensity_LN_vs_V_log_V.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Predicted_Peak_Intensity_LN_vs_Type_II_Area_Symlog_Scale.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Predicted_Peak_Intensity_LN_vs_Diffusive_Shock_Log_Scale.png",
                         "--png",
                            "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Predicted_Peak_Intensity_LN_vs_2nd_order_speed_at_20_solar_radii.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Predicted_Peak_Intensity_LN_vs_Half_Width.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Predicted_Peak_Intensity_LN_vs_Acceleration_Symlog_Scale.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Predicted_Peak_Intensity_LN_vs_Latitude.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Predicted_Peak_Intensity_LN_vs_Linear_Speed.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Predicted_Peak_Intensity_LN_vs_Longitude.png"

                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\richardson_mixed\\retrained_autoencoder_ss_0_5\\Scaled_Predictions_F1_0.6666666666666666.png",


                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Score_vs_Number_of_CMEs_in_the_Past_Month.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Score_vs_Half_Width.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Score_vs_Acceleration_Symlog_Scale.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Score_vs_Latitude.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Score_vs_Linear_Speed.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\features\\Score_vs_Longitude.png"


                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\classifier_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\dl_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\rc_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\re_F1_vs_Oversampling_Rate.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\regression_F1_vs_Oversampling_Rate.png",

                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\cRegNN_20_median.png",
                         #"--png",
                         #   "C:\\Users\\Daniel\\source\\repos\\Thesis_without_double_cme_column\\Thesis_fix_normalization\\eval\\crt_60_median.png",
                        ])
    main_args(args)