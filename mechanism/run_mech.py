import os
"""
for flag in ["original","feedback_only","wo_feedback"]:
    os.system(f"python -u internalmech.py --benchmark realtoxicity --selfcorr_flag feedback --feedback_flag {flag} --target_file /home/zhiyu2/guangliang/SuperficialMoralSelfCorrection/mechanism/naaclresults/RealToxicity/extrinsic.json")
for flag in ["original","feedback_only","wo_feedback"]:
    os.system(f"python -u internalmech.py --benchmark bbq --selfcorr_flag feedback --feedback_flag {flag} --target_file /home/zhiyu2/guangliang/SuperficialMoralSelfCorrection/mechanism/naaclresults/Sexual_orientation/extrinsic.json")
    """

os.system("python -u internalmech.py --benchmark realtoxicity --selfcorr_flag feedback-CoT  --target_file /home/zhiyu2/guangliang/SuperficialMoralSelfCorrection/mechanism/naaclresults/RealToxicity/extrinsic_cot.json")
os.system("python -u internalmech.py --benchmark bbq --selfcorr_flag feedback-CoT  --target_file /home/zhiyu2/guangliang/SuperficialMoralSelfCorrection/mechanism/naaclresults/Sexual_orientation/extrinsic_cot.json")
os.system("python -u internalmech.py --benchmark realtoxicity --selfcorr_flag intrinsic-feedback-CoT  --target_file /home/zhiyu2/guangliang/SuperficialMoralSelfCorrection/mechanism/naaclresults/RealToxicity/intrinsic_extrinsic_cot.json")
os.system("python -u internalmech.py --benchmark bbq --selfcorr_flag intrinsic-feedback-CoT  --target_file /home/zhiyu2/guangliang/SuperficialMoralSelfCorrection/mechanism/naaclresults/Sexual_orientation/intrinsic_extrinsic_cot.json")