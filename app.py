
# import cv2
from flask import jsonify

import numpy as  np

import shutil

import random
import os
# import psycopg2
from flask import Flask, render_template, request, url_for, redirect,Response
from flask_cors import CORS, cross_origin

from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from PIL import Image, ImageOps




# import for image segmentation
import models
from models.convert_pidinet import convert_pidinet
from edge_dataloader import  BSDS_Loader
from torch.utils.data import DataLoader
import os
import time
import torch
import torchvision
import numpy as np
import cv2
from utils import *

# for encryption-decryption
import security
import config
# imports the config.py file containing patient and doctor login info
import send_email


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# important values to be set
class Args:
    def __init__(self):
        self.ablation=False
        self.savedir='./static/table5_our'
        self.use_cuda=False
        self.model='pidinet_converted', 

        self.config='carv4'
        self.sa=True
        self.dil=True  
        self.evaluate_converted=True
#         self.datadir='./data/BSDS500cubs'
        self.datadir='./data/BSDS500Single'
        self.eta=0.3
        self.evaluate='./data/table5_our/save_models/checkpoint_007_.pth'
args=Args()        



# extra functions for line width


def test(test_loader, model, epoch, running_file,args):

    print("Args are ",args)

    from PIL import Image
    import scipy.io as sio
    model.eval()

    if args.ablation:
        img_dir = os.path.join(args.savedir, 'eval_results_val', 'imgs_epoch_%03d' % (epoch - 1))
        mat_dir = os.path.join(args.savedir, 'eval_results_val', 'mats_epoch_%03d' % (epoch - 1))
    else:
        img_dir = os.path.join(args.savedir, 'eval_results', 'imgs_epoch_%03d' % (epoch - 1))
        mat_dir = os.path.join(args.savedir, 'eval_results', 'mats_epoch_%03d' % (epoch - 1))
    eval_info = '\nBegin to eval...\nImg generated in %s\n' % img_dir
    print(eval_info)
    running_file.write('\n%s\n%s\n' % (str(args), eval_info))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    else:
        print('%s already exits, yes' % img_dir)
        #return
    if not os.path.exists(mat_dir):
        print("Creating folder",mat_dir)
        os.makedirs(mat_dir)
    average_thicknesses=[]
    for idx, (image, img_name) in enumerate(test_loader):
        print("Going for test loader batch",idx)
        img_name = img_name[0]
        with torch.no_grad():
            image = image.cuda() if args.use_cuda else image
            _, _, H, W = image.shape
            results = model(image)
            result = torch.squeeze(results[-1]).cpu().numpy()

        results_all = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
          results_all[i, 0, :, :] = results[i]
        print("Completed one batch")
        torchvision.utils.save_image(1-results_all, 
                os.path.join(img_dir, "%s.jpg" % img_name))
        sio.savemat(os.path.join(mat_dir, '%s.mat' % img_name), {'img': result})
        result = Image.fromarray((result * 255).astype(np.uint8))
        print(np.asarray(result))

        # find the white line

        mask_array=create_mask(np.asarray(result))
        mask_image = Image.fromarray(mask_array.astype(np.uint8))
        mask_image_name=os.path.join(img_dir, "%s_mask.png" % img_name)
        mask_image.save(mask_image_name)
        print("mask saved at ",mask_image_name)

        up_points,down_points=find_polyline_y(mask_array)
        up_points,down_points=find_best_polyline_y(up_points,down_points)




        print("top left",up_points[0])
        print("bottom_right",down_points[-1])


        


        result.save(os.path.join(img_dir, "%s.png" % img_name))
        image = cv2.imread(os.path.join(img_dir, "%s.png" % img_name))
        color = (0, 0, 255)
        fcolor = (255, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1


        # Line thickness of 2 px
        thickness = 3
        calculated_thickness=[]
        for i in range(up_points.shape[0]):
            up_point=(up_points[i][1],up_points[i][0])
            down_point=(down_points[i][1],down_points[i][0])
            calculated_thickness.append(down_points[i][0]-up_points[i][0])
            image=cv2.line(image, up_point, down_point, color, thickness) 
        average_thickness=round(np.mean(np.array(calculated_thickness)),2)
        image = cv2.putText(image, "th:"+str(average_thickness), (up_points[0][1]-15,up_points[0][0]-15), font, 
                   fontScale, fcolor, thickness, cv2.LINE_AA)

        average_thicknesses.append(average_thickness)
        cv2.imwrite(os.path.join(img_dir, "%s_box.jpeg" % img_name),image)

        # result_array=np.asarray(image)
        # print("result shsape is ",result_array.shape)


        runinfo = "Running test [%d/%d]" % (idx + 1, len(test_loader))
        print(runinfo)
        running_file.write('%s\n' % runinfo)
    running_file.write('\nDone\n')
    return average_thicknesses,mask_image_name

def create_mask(arr):
   max_val=np.max(arr)
   mask_threshold=int(0.975*max_val)
   print("mask_threshold",mask_threshold)
#     mask_threshold=200
   new_matrix=np.zeros(arr.shape)
   new_matrix[arr>mask_threshold]=255
   return new_matrix

def find_polyline_y(mask_array):
    up_points=[]
    down_points=[]    
    
    for col_num in range(mask_array.shape[1]):
        for row_num in range(mask_array.shape[0]):
            if mask_array[row_num][col_num]==255:
                break

        up_point=(row_num,col_num)
        for row_num in range(row_num+1,mask_array.shape[0]):
            if mask_array[row_num][col_num]==0:
                break
        down_point=(row_num-1,col_num)

        if down_point[0]>up_point[0]+1:
            up_points.append(up_point)
            down_points.append(down_point)


    return up_points,down_points


def find_best_polyline_y(up_points,down_points):
    up_points=np.array(up_points)
    down_points=np.array(down_points)

    

    mean_up_row_num=int(np.mean(up_points[:,0]))
    mean_down_row_num=int(np.mean(down_points[:,0]))
    to_remove=[]
    for i in range(up_points.shape[0]):
        if not(up_points[i][0]>mean_up_row_num-50 and up_points[i][0]<mean_up_row_num+50):
            to_remove.append(i)

    up_points=np.delete(up_points,to_remove,axis=0)
    down_points=np.delete(down_points,to_remove,axis=0)

    print(mean_up_row_num,mean_down_row_num)
    return up_points,down_points


def get_thickness_image(im):
    im_name='im1.jpeg'
    im_just_name='im1'
    save_path=os.path.join('data','BSDS500Single','my_data','test_img',im_name)
    write_path=str(os.path.join('my_data','test_img',im_name))
    lst_file_path=os.path.join('data','BSDS500Single','test.lst')
    with open(lst_file_path,"w") as f:
        f.write(write_path)
    im.save(save_path)

    model=models.my_pidinet_converted(args.config,args.sa,args.dil)
    running_file = os.path.join(args.savedir, '%s_running-%s.txt' \
            % (args.model, time.strftime('%Y-%m-%d-%H-%M-%S')))

    with open(running_file, 'w+') as f:
        checkpoint = load_checkpoint(args, f)
    model.load_state_dict(convert_pidinet(checkpoint['state_dict'], args.config))    
    test_dataset = BSDS_Loader(root=args.datadir, split="test", threshold=args.eta)
    test_loader = DataLoader(test_dataset, batch_size=12, num_workers=0, shuffle=False)
    epoch=8
    with open(running_file, 'w+') as f:
        all_thicknesses_avg,mask_img_loc=test(test_loader, model, epoch, f,args)

    print("thicknesss is ",all_thicknesses_avg)
    return all_thicknesses_avg[0],im_just_name,mask_img_loc




import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
def pre_image(image_path,model):
    '''
    predictions on a single masked image
    '''
    im = Image.open(image_path)
    img = im.convert("RGB")
    # im_arr=np.array(im)
    
    # im_arr=im_arr[:,:,0:3]
    # img = Image.fromarray(im_arr)
#     print("original shape is ",np.array(img).shape)


    
    transform_norm = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # get normalized image
    img_normalized = transform_norm(img).float()
#     print("1",img_normalized.shape)    
    img_normalized = img_normalized.unsqueeze_(0)
    print(img_normalized.shape)    
    # input = Variable(image_tensor)
    img_normalized = img_normalized.to(device)
#     print(img_normalized.shape)
    with torch.no_grad():
        model.eval()  
        output =model(img_normalized)
        index = output.data.cpu().numpy().argmax()
        classes = [0,1]
        class_name = classes[index]
        return class_name

def load_classifier():
    model_ft = torchvision.models.efficientnet_b0(pretrained=True)
    model_ft.classifier[1] = nn.Linear(1280, 2)
    model_ft = model_ft.to(device)
    num_epochs=10
    model_state_path="classifier_models/simple_efficientnet_seg_ultra_gpu_"+str(num_epochs)+"_wtRecord.pt"
    model_ft.load_state_dict(torch.load(model_state_path,map_location=device))
    return model_ft


app = Flask(__name__)

cors = CORS(app)
PEOPLE_FOLDER = os.path.join('static', 'ultras')

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def index():
    return "okay"

@app.route('/patientLogin/', methods=('GET','POST'))
def patientLogin():
    patients=config.patients

    if request.method == 'POST':
        email = str(request.form['email'])
        pwd = str(request.form['pwd'])

        if email not in patients:
            return "Email ID does not exist"
        else:
            login_pass=patients[email]
            pwd=pwd.strip()
            if pwd!=login_pass:
                return "Password incorrect"
            else:
                print("Correct")

                # return render_template("upload.html")
                return redirect(url_for("upload"))



        # im_invert = ImageOps.invert(im)
        
        # full_filename = os.path.join('static','ultras', fname)
        # part_filename = os.path.join('ultras', fname)

        # im_invert.save(full_filename, quality=95)

        
        
        
    print("This is get")
    return render_template('patientLogin.html')

@app.route('/allPatients/', methods=('GET','POST'))
def allPatients():

    page_details={}

    all_my_patients_details=[]
    source_file_location="static/saved_patient_images/"
    for f in os.listdir(source_file_location):
        if ".DS_Store" in f:
            continue
        print("file is ",f)
        f_name_without_extension=f.split(".")[0]
        f_name_parts=f_name_without_extension.split("_")
        print(f_name_parts)
        # let us try to re formulate the email ID
        username=f_name_parts[6]
        # get the rest
        rest_of_email=""
        for i in range(7,len(f_name_parts)):
            print(f_name_parts[i])
            rest_of_email=rest_of_email+f_name_parts[i]+"."


        
        
        full_email=username+"@"+rest_of_email
        full_email=full_email[:-1]
        # print(full_email,email)
        # if full_email==email:
        page_details["doc_name"]=username                        
        # this patient is under me
        this_patient={}
        this_patient["first_name"]=security.encrypt(plaintext=f_name_parts[1])
        this_patient["last_name"]=security.encrypt(plaintext=f_name_parts[2])
        this_patient["age"]=security.encrypt(plaintext=f_name_parts[3])
        this_patient["tel_num"]=security.encrypt(plaintext=f_name_parts[4])
        this_patient["normality"]=f_name_parts[5]
        this_patient["file_location"]="saved_patient_images/"+f
        all_my_patients_details.append(this_patient)




    page_details["all_my_patients_details"]=all_my_patients_details
    # return render_template("upload.html")
    return render_template("myPatients.html", page_details = page_details)



@app.route('/drLogin/', methods=('GET','POST'))
def drLogin():
    doctors=config.doctors

    if request.method == 'POST':
        email = str(request.form['email'])
        pwd = str(request.form['pwd'])

        if email not in doctors:
            return "Email ID does not exist"
        else:
            login_pass=doctors[email]
            pwd=pwd.strip()
            if pwd!=login_pass:
                return "Password incorrect"
            else:
                print("Correct")
                # get all my patients details
                page_details={}

                all_my_patients_details=[]
                source_file_location="static/saved_patient_images/"
                for f in os.listdir(source_file_location):
                    if ".DS_Store" in f:
                        continue
                    print("file is ",f)
                    f_name_without_extension=f.split(".")[0]
                    f_name_parts=f_name_without_extension.split("_")
                    print(f_name_parts)
                    # let us try to re formulate the email ID
                    username=f_name_parts[6]
                    # get the rest
                    rest_of_email=""
                    for i in range(7,len(f_name_parts)):
                        print(f_name_parts[i])
                        rest_of_email=rest_of_email+f_name_parts[i]+"."


                    
                    
                    full_email=username+"@"+rest_of_email
                    full_email=full_email[:-1]
                    print(full_email,email)
                    if full_email==email:
                        page_details["doc_name"]=username                        
                        # this patient is under me
                        this_patient={}
                        this_patient["first_name"]=f_name_parts[1]
                        this_patient["last_name"]=f_name_parts[2]
                        this_patient["age"]=f_name_parts[3]
                        this_patient["tel_num"]=f_name_parts[4]
                        this_patient["normality"]=f_name_parts[5]
                        this_patient["file_location"]="saved_patient_images/"+f
                        all_my_patients_details.append(this_patient)




                page_details["all_my_patients_details"]=all_my_patients_details
                # return render_template("upload.html")
                return render_template("myPatients.html", page_details = page_details)



        # im_invert = ImageOps.invert(im)
        
        # full_filename = os.path.join('static','ultras', fname)
        # part_filename = os.path.join('ultras', fname)

        # im_invert.save(full_filename, quality=95)

        
        
        
    print("This is get")
    return render_template('drLogin.html')    





@app.route('/upload/', methods=('GET','POST'))
def upload():
    if request.method == 'POST':
        first_name = str(request.form['first_name'])
        last_name = str(request.form['last_name'])
        tel_num = str(request.form['tel_num'])
        age = str(request.form['age'])
        # dob = str(request.form['dob'])
        dremail=str(request.form['dremail'])
        original_dremail=dremail
        dremail=dremail.replace("@","_")
        dremail=dremail.replace(".","_")
        f = request.files['avatar']
        

        im = Image.open(f)
        thickness,im_just_name,mask_img_loc=get_thickness_image(im)
        print("mask image at ",mask_img_loc)

        model_ft=load_classifier()
        normality=pre_image(mask_img_loc,model_ft)
        if normality==0:
            normality="Normal"
        else:
            normality="Abnormal"

        

        print("Image name is ",im_just_name)

        # full_filename = os.path.join('data','table5_our','eval_results','imgs_epoch_007', im_just_name+"_box.jpeg")
        part_filename = os.path.join('table5_our','eval_results','imgs_epoch_007', im_just_name+"_box.jpeg")

        print("part file name",part_filename)
        
        full_filename = os.path.join('static',part_filename)

        print("full full_filename",full_filename)

        im_just_name=im_just_name+"_"+first_name
        im_just_name=im_just_name+"_"+last_name
        im_just_name=im_just_name+"_"+age        
        im_just_name=im_just_name+"_"+tel_num
        im_just_name=im_just_name+"_"+normality
        im_just_name=im_just_name+"_"+dremail
        im_just_name=im_just_name+".jpeg"
        target_path=os.path.join("static","saved_patient_images",im_just_name)

        print(im_just_name,"to be saved at",target_path)   

        # also save the image somewhere 
        shutil.copy(full_filename,target_path)





        

        user_data={}
        user_data["part_filename"]=part_filename
        user_data["first_name"]=first_name
        user_data["thickness"]=str(thickness)
        user_data["normality"]=normality
        user_data["dremail"]=original_dremail


        subject = "USG Test results for "+first_name+" "+last_name
        body = "Dear Doctor,\n"
        body=body+"Please find attached the USG report of "+first_name+" "+last_name+"\n"
        body=body+"Thickness = "+str(thickness)+"\n"
        body=body+"Status: "+normality+"\n\n"
        body=body+"Contact "+first_name+" at: "
        body=body+tel_num+"\n\n"
        body=body+"Regards,\nTeamQUAnalytics"
        sender = "sdp23.mm@gmail.com"
        recipients = [original_dremail]
        password = "xpwhbsfhoszdyubf"
        print("Sending email")
        send_email.go_mail(subject, body, target_path, sender, recipients, password)        
        print("Email sent")
        
        
        
        
        return render_template("usgresult.html", user_data = user_data)
        
    print("This is get")
    return render_template('upload.html')

    






if __name__=="__main__":
    app.run(host="0.0.0.0",port=5001,debug=True)    
    # app.run(port=5001)    