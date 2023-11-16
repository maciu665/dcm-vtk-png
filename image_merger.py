import cv2
import numpy as np
import sys
import os
from PIL import Image, ImageDraw, ImageFont
'''
172
1 - porównania tego samego przekroju tego samego urazu w wielu różnych użytych C i W aby pokazać jak c i w wpływają na widoczność
[13:03]
2 - urazy z 4 grup urazów na 4 różnych C i W - w siatce 4x4 - aby pokazać jak różne c i w pasują tylko do jedej grupy urazu
[13:04]
3 - obrazek z wizualizacją 3D czaszki skóry itp i pokazanym w to wklejonym przekrojem z dicom aby pokazać jak one leżą
[13:07]
4 - porównanie dwóch sąsiednich przekrojów i oblicznei różnicy między nimi - aby pokazać dlaczego trzeba rozsądnie podejść do doboru plików testowych i treningowych aby uniknąć przeuczenia  - to też warto opisać w tekście
[13:08]
5 - mozna spróbować, ale to bedzie bardiej skomplikowane, nanieść na widok 3D predykcje modelu



# kolaz 4 x 4

mats = []
pods = []
wrls = []

cell_size = [640,640]
imagedir = "/home/maciejm/PHD/BRAINATOR/SPRAWDZONE/PNG/"
new_image = Image.new(size=(cell_size[0]*4+64,cell_size[1]*4+64),mode="RGBA",color=(255,255,255,255))
rot_image = Image.new(size=(cell_size[0]*4+64,cell_size[1]*4+64),mode="RGBA",color=(255,255,255,0))

draw = ImageDraw.Draw(new_image)
draw2 = ImageDraw.Draw(rot_image)
font2 = ImageFont.truetype("FreeMonoBold.ttf", 80)

for nc,c in enumerate([-100,0,100,300]):
    for wc,w in enumerate([100,250,500,1000]):
        imname = "C%i-W%i"%(c,w)
        print(imname)
        filename = os.path.join(imagedir,imname,"17","Z","slice_172.png")
        print(filename)
        temp_image = Image.open(filename)
        new_image.paste(temp_image,(cell_size[0]*nc+64+64,cell_size[1]*wc+64+64))
        if wc == 0:
            draw.text((cell_size[0]*nc+200,cell_size[1]*wc+20),"C = %i"%c,(0,0,0,255),font=font2)
        if nc == 0:
            draw2.text((cell_size[0]*(3-wc)+170,cell_size[1]*nc+20),"W = %i"%w,(0,0,0,255),font=font2)

rot_image = rot_image.rotate (90, expand = 1)
new_image.paste(rot_image,mask=rot_image)

new_image.save("/home/maciejm/PHD/BRAINATOR/C_W_COMBINATIONS.png")
# rot_image.save("/home/maciejm/PHD/BRAINATOR/C_W_COMBINATIONS.png")



mats = []
pods = []
wrls = []

cell_size = [640,640]
imagedir = "/home/maciejm/PHD/BRAINATOR/SPRAWDZONE/PNG/"

new_image = Image.new(size=(cell_size[0]*4+64,cell_size[1]*4+64+64),mode="RGBA",color=(255,255,255,255))
rot_image = Image.new(size=(cell_size[0]*4+64,cell_size[1]*4+64+64),mode="RGBA",color=(255,255,255,0))

draw = ImageDraw.Draw(new_image)
draw2 = ImageDraw.Draw(rot_image)
font2 = ImageFont.truetype("FreeMonoBold.ttf", 80)

urazy = ["MO","ZX","MK","SDH"]
cases = [1,2,4,12]
slices = [366,22,146,179]
cws = [[40,350],[350,3500],[40,100],[33,102]]
reds = []

for row in range(4):
    for col in range(4):
        tifdir = os.path.join(imagedir.replace("PNG","TIF"),str(cases[row]))
        tlist = os.listdir(tifdir)
        for i in tlist:
            # print(i)
            if i.startswith(str(slices[row])+".") and i.endswith("png"):
                redfile = os.path.join(tifdir,i)
                break
        redimg = Image.open(redfile)
        print(redimg)

        r,g,b = redimg.split()
        ar = np.array(r)
        ag = np.array(g)
        where = np.where((ar == 255) & (ag == 0))
       
        imname = "C%i-W%i"%(cws[col][0],cws[col][1])
        print(imname)
        filename = os.path.join(imagedir,imname,str(cases[row]),"Z","slice_%s.png"%str(slices[row]).zfill(3))
        print(filename)
        temp_image = Image.open(filename)
        draw3 = ImageDraw.Draw(temp_image)
        for i in range(where[0].shape[0]):
            x = where[0][i]
            y = where[1][i]
            # sys.exit()
            if col == row:
                draw3.ellipse((y-2,x-2,y+2,x+2), fill=(0,255,0),outline=(0,255,0))
            else:
                draw3.ellipse((y-2,x-2,y+2,x+2), fill=(255,0,0),outline=(255,0,0))
        # if slices[row] == 146 and col == row:
            # temp_image.save("/home/maciejm/PHD/BRAINATOR/146.png")
            # sys.exit()
        

        new_image.paste(temp_image,(cell_size[0]*col+64+64,cell_size[1]*row+64+64+64))
        if row == 0:
            draw.text((cell_size[0]*col+200,cell_size[1]*row+20),"C = %i\nW = %i"%(cws[col][0],cws[col][1]),(0,0,0,255),font=font2)
        if col == 0:
            draw2.text((cell_size[0]*(3-row)+220,cell_size[1]*col+20),urazy[row],(0,0,0,255),font=font2)

rot_image = rot_image.rotate (90, expand = 1)
new_image.paste(rot_image,mask=rot_image)

new_image.save("/home/maciejm/PHD/BRAINATOR/C_W_INJURIES.png")
# rot_image.save("/home/maciejm/PHD/BRAINATOR/C_W_COMBINATIONS.png")

'''

import matplotlib
import matplotlib.cm as cm

cell_size = [512,512]
imagedir = "/home/maciejm/GIT/dcm-vtk-png/"

new_image = Image.new(size=((cell_size[0]+128)*4,cell_size[1]*2+512),mode="RGBA",color=(255,255,255,255))

draw = ImageDraw.Draw(new_image)
font2 = ImageFont.truetype("FreeMonoBold.ttf", 48)

imagelist = []
leglist = []
nazwy = {}

nazwy["bw"] = "black-white"
nazwy["cw"] = "cool-warm"
nazwy["br"] = "blue-red\nrainbow"
nazwy["bwsel"] = "black-white\n+selection"
nazwy["bwnl"] = "black-white\nbilinear"
nazwy["bwba"] = "black-white\n+below/above range"
nazwy["bands"] = "arbitrary bands"
imageset = ['bw','cw','br','bwsel','bwnl','bwba','bands']

for pname in imageset:
    imagelist.append(Image.open(os.path.join(imagedir,"slice_172-%s.png"%pname)))
    leglist.append(Image.open(os.path.join(imagedir,"mapa-%s.png"%pname)))

for i in range(7):
    print(i//4,i%4)
    x = (cell_size[0]+128)*(i%4)+64+int((i//4)*(cell_size[0]+128)*0.5)
    y = 128+(cell_size[1]+256)*(i//4)
    new_image.paste(imagelist[i],(x,y))
    new_image.paste(leglist[i],(x,y+cell_size[1]))
    bbox = draw.textbbox((0,0),text=nazwy[imageset[i]],font=font2)
    draw.text((x+int(cell_size[0]*0.5)-int(0.5*bbox[2]),y-64-int(0.5*bbox[3])),nazwy[imageset[i]],(0,0,0,255),font=font2,align="center")

new_image.save(os.path.join(imagedir,"palettes.png"))

sys.exit()

norm = matplotlib.colors.Normalize(vmin=95, vmax=159, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
norm2 = matplotlib.colors.Normalize(vmin=0, vmax=255, clip=True)
mapper2 = cm.ScalarMappable(norm=norm2, cmap=cm.gray)

scalebw = []
scalebr = []

for i in range(255,0,-1):
    scalebw.append(np.full(128,i))
    scalebw.append(np.full(128,i))

for i in range(159,95,-1):
    for j in range(8):
        scalebr.append(np.full(128,i))

for i in range(1,4):
    image0 = imagelist[i-1]
    image1 = imagelist[i]
    r0,g,b = image0.split()
    r1,g,b = image1.split()
    ar0 = np.array(r0,dtype="int16")
    ar1 = np.array(r1,dtype="int16")

    difar = np.subtract(ar1,ar0)
    print(difar.dtype)
    difar = difar.reshape((512*512))
    print(np.min(difar),np.max(difar))
    # difar = np.multiply(difar,4)
    # difar =
    # sys.exit()

    p128 = np.full((512*512), 128)
    reds = []
    greens = []
    blues = []
    adata = np.add(difar,p128)
    # print(adata)
    # sys.exit()


    color = mapper.to_rgba(adata)
    print(color.shape)
    argb = color.reshape((512,512,4))
    print(argb.shape)
    # print(argb)
    im = Image.fromarray((argb * 255).astype(np.uint8))
    new_image.paste(im,(cell_size[0]*i-320+22,720))

    draw.line([cell_size[0]*i-320-32,512+64,cell_size[0]*i-32,720], width = 4, fill=(0,0,0))
    draw.line([cell_size[0]*i+320-32,512+64,cell_size[0]*i-32,720], width = 4, fill=(0,0,0))

colorbw = mapper2.to_rgba(scalebw)
print(colorbw.shape)
im = Image.fromarray((colorbw * 255).astype(np.uint8))
new_image.paste(im,(32,720))

colorbr = mapper.to_rgba(scalebr)
print(colorbr.shape)
im = Image.fromarray((colorbr * 255).astype(np.uint8))
new_image.paste(im,(2400-96+22,720))


draw.text((144+22,720-32),"255",(0,0,0,255),font=font2)
draw.text((144+22,720-32+512),"0",(0,0,0,255),font=font2)
draw.text((2224+22,720-32),"32",(0,0,0,255),font=font2)
draw.text((2184+22,720-32+512),"-32",(0,0,0,255),font=font2)

draw.rectangle([32,720,32+128,720+512], width = 2, outline=(0,0,0))
draw.rectangle([2326,720,2304+150,720+512], width = 2, outline=(0,0,0))

        

new_image.save("/home/maciejm/PHD/BRAINATOR/DCM_DIFFERENCE.png")

'''

sys.exit()

lista = os.listdir(imagedir)
lista.sort()
for file in lista:
    if file.startswith("mat"):
        mats.append(os.path.join(imagedir,file))
    if file.startswith("pod"):
        pods.append(os.path.join(imagedir,file))
    if file.startswith("wrl"):
        wrls.append(os.path.join(imagedir,file))

cols = 4
rows = 4
cell_size = [720,760]

print(mats)

new_image = Image.new(size=(cols*cell_size[0],rows*cell_size[1]),mode="RGB",color=(255,255,255))

obrazy = [[mats,0,4],[pods,0,4],[pods,4,3],[wrls,0,3]]

draw = ImageDraw.Draw(new_image)
font2 = ImageFont.truetype("FreeMonoBold.ttf", 80)
# 
letters = "abcdefghijklmnopqrst"
letter = 0

for row in range(rows):
    for col in range(cols):
        offset = obrazy[row][1]
        count = obrazy[row][2]
        print(row, col, offset, count)
        if col < count:
            temp_image_file = obrazy[row][0][col+offset]
            offset = obrazy[row][1]
            print(temp_image_file)
            #sys.exit()
            temp_image = Image.open(temp_image_file)
            new_image.paste(temp_image,(cell_size[0]*col+10,cell_size[1]*row+100))
            draw.text((cell_size[0]*col+300,cell_size[1]*row+10),letters[letter],(0,0,0),font=font2)
            letter += 1


#draw.line([10,760,cell_size[0]*cols-70,760], width = 4, fill=(0,0,0))
#draw.line([10,760+2*760,cell_size[0]*(cols-1)-70,760+2*760], width = 4, fill=(0,0,0))

new_image.save("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/KOMPLET_MATS3.png")




imdir0 = "/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/test_miss0/"
imdir1 = "/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/test_miss1/"

outdir = "/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/test_miss/"

lista = os.listdir(imdir0)

maska = Image.open("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/maska299.png")

for image_name in lista:
    image0 = Image.open(os.path.join(imdir0,image_name))
    image1 = Image.open(os.path.join(imdir1,image_name))

    image0.paste(image1, (0, 0), maska)

    draw = ImageDraw.Draw(image0)
    draw.line((0, 0) + image0.size, fill=(255,0,255),width = 2)

    image0.save(os.path.join(outdir,image_name)) 


# kolaz 6 x 3

cols = 6
rows = 3
cell_size = [360,360]

new_image = Image.new(size=(cols*cell_size[0],rows*cell_size[1]),mode="RGB",color=(255,255,255))

imdirWB = "/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/test_missWB/"
imdirSB = "/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/test_missSB/"
imdirMIX = "/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/test_miss/"

listWB = os.listdir(imdirWB)
listSB = os.listdir(imdirSB)
listMIX = os.listdir(imdirMIX)
imdirs = [imdirWB,imdirSB,imdirMIX] 
lists = [listWB,listSB,listMIX]

for row in range(rows):
    for col in range(cols):
        temp_image = Image.open(os.path.join(imdirs[row],lists[row][col]))
        new_image.paste(temp_image,(cell_size[0]*col+30,cell_size[1]*row+30))

new_image.save("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/missMIX.png")


sys.exit()


# kolaz 8 x 5

classes = ["C-BEAM","FLAT-BAR","HOLLOW-SECTION","I-BEAM","L-BEAM","PIPE","ROUND-BAR","SQUARE-BAR"]
# styles = ["C1","C2","C3","FB","HB","OB","RB","RN","RS","SB","WB"]
# styles = ["WB","SB","RS","RN","RB"]
styles = ["WIREFRAME_BLACK_BG","SHADED_BLACK_BG","RENDER_SIMPLE","RENDER_PHOTO_NO_BG","RENDER_PHOTO_BG"]
styles = ["HIDDEN_BLACK_BG","OVERLAY_BLACK_BG","FEATURE_BLACK_BG","WNOAA_BLACK_BG","SNOAA_BLACK_BG"]
styles = ["WNOAA_BLACK_BG","SNOAA_BLACK_BG"]
styles = ["HIDDEN_BLACK_BG","OVERLAY_BLACK_BG","FEATURE_BLACK_BG"]

cols = 8
rows = 3
cell_size = [320,340]

new_image = Image.new(size=(cols*cell_size[0],rows*cell_size[1]),mode="RGB",color=(255,255,255))

for row in range(rows):
    for col in range(cols):
        temp_image_file = "/home/maciejm/PHD/PUBLIKACJA_02/images3/res299/%s/%s/%s_00028-view01.png"%(styles[row],classes[col],classes[col])
        print(temp_image_file)
        #sys.exit()
        temp_image = Image.open(temp_image_file)
        new_image.paste(temp_image,(cell_size[0]*col+10,cell_size[1]*row+20))

new_image.save("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/STYLE8-11.png")



#img1 = cv2.imread('img1.png')
#img2 = cv2.imread('img2.png')



#vis = np.concatenate((img1, img2), axis=1)
#cv2.imwrite('out.png', vis)



# powiekszenia
classes = ["C-BEAM","FLAT-BAR","HOLLOW-SECTION","I-BEAM","L-BEAM","PIPE","ROUND-BAR","SQUARE-BAR"]

wbaa_image = Image.open("/home/maciejm/PHD/PUBLIKACJA_02/images3/res299/WIREFRAME_BLACK_BG/I-BEAM/I-BEAM_00002-view02.png")
wbna_image = Image.open("/home/maciejm/PHD/PUBLIKACJA_02/images3/res299/WNOAA_BLACK_BG/I-BEAM/I-BEAM_00002-view02.png")
sbaa_image = Image.open("/home/maciejm/PHD/PUBLIKACJA_02/images3/res299/SHADED_BLACK_BG/I-BEAM/I-BEAM_00002-view02.png")
sbna_image = Image.open("/home/maciejm/PHD/PUBLIKACJA_02/images3/res299/SNOAA_BLACK_BG/I-BEAM/I-BEAM_00002-view02.png")
images = [wbaa_image,wbna_image,sbaa_image,sbna_image]

#wbaa_image_crop = wbaa_image.crop((135, 135, 185, 185)).resize((240,240), Image.Resampling.)
crops = []
for temp_image in images:
    crops.append(temp_image.crop((135, 135, 165, 165)).resize((240,240), Image.Resampling.NEAREST))

cols = 4
rows = 2
cell_size = [400,400]

new_image = Image.new(size=(cols*cell_size[0],rows*cell_size[1]),mode="RGB",color=(255,255,255))
draw = ImageDraw.Draw(new_image)

for col in range(cols):
    new_image.paste(images[col],(cell_size[0]*col+50,cell_size[1]*1+50))
    new_image.paste(crops[col],(cell_size[0]*col+80,cell_size[1]*0+80))
    
    draw.rectangle([cell_size[0]*col+50+135,cell_size[1]*1+50+135,cell_size[0]*col+50+165,cell_size[1]*1+50+165], width = 2, outline=(255,0,255))
    draw.rectangle([cell_size[0]*col+80,cell_size[1]*0+80,cell_size[0]*col+320,cell_size[1]*0+320], width = 4, outline=(255,0,255))

    draw.line([cell_size[0]*col+80,cell_size[1]*0+320,cell_size[0]*col+185,cell_size[1]*1+50+165], width = 2, fill=(255,0,255))
    draw.line([cell_size[0]*col+320,cell_size[1]*0+320,cell_size[0]*col+185+30,cell_size[1]*1+50+165], width = 2, fill=(255,0,255))


new_image.save("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/NOAA.png")



# kolaz 4 x 4

mats = []
pods = []
wrls = []

imagedir = "/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY"

lista = os.listdir(imagedir)
lista.sort()
for file in lista:
    if file.startswith("mat"):
        mats.append(os.path.join(imagedir,file))
    if file.startswith("pod"):
        pods.append(os.path.join(imagedir,file))
    if file.startswith("wrl"):
        wrls.append(os.path.join(imagedir,file))

cols = 4
rows = 4
cell_size = [720,760]

print(mats)

new_image = Image.new(size=(cols*cell_size[0],rows*cell_size[1]),mode="RGB",color=(255,255,255))

obrazy = [[mats,0,4],[pods,0,4],[pods,4,3],[wrls,0,3]]

draw = ImageDraw.Draw(new_image)
font2 = ImageFont.truetype("FreeMonoBold.ttf", 80)
# 
letters = "abcdefghijklmnopqrst"
letter = 0

for row in range(rows):
    for col in range(cols):
        offset = obrazy[row][1]
        count = obrazy[row][2]
        print(row, col, offset, count)
        if col < count:
            temp_image_file = obrazy[row][0][col+offset]
            offset = obrazy[row][1]
            print(temp_image_file)
            #sys.exit()
            temp_image = Image.open(temp_image_file)
            new_image.paste(temp_image,(cell_size[0]*col+10,cell_size[1]*row+100))
            draw.text((cell_size[0]*col+300,cell_size[1]*row+10),letters[letter],(0,0,0),font=font2)
            letter += 1


#draw.line([10,760,cell_size[0]*cols-70,760], width = 4, fill=(0,0,0))
#draw.line([10,760+2*760,cell_size[0]*(cols-1)-70,760+2*760], width = 4, fill=(0,0,0))

new_image.save("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/KOMPLET_MATS3.png")

#img1 = cv2.imread('img1.png')
#img2 = cv2.imread('img2.png')



#vis = np.concatenate((img1, img2), axis=1)
#cv2.imwrite('out.png', vis)


# kolaz 4 x 4 RGB

ob_image = "/home/maciejm/PHD/PUBLIKACJA_02/images3/res299/SHADED_BLACK_BG/I-BEAM/I-BEAM_00006-view02.png"
c1_image = "/home/maciejm/PHD/PUBLIKACJA_02/images3/res299/COMP1_BLACK_BG/I-BEAM/I-BEAM_00006-view02.png"
c2_image = "/home/maciejm/PHD/PUBLIKACJA_02/images3/res299/COMP2_BLACK_BG/I-BEAM/I-BEAM_00006-view02.png"
c3_image = "/home/maciejm/PHD/PUBLIKACJA_02/images3/res299/COMP3_BLACK_BG/I-BEAM/I-BEAM_00006-view02.png"
c4_image = "/home/maciejm/PHD/PUBLIKACJA_02/images3/res299/COMP4_BLACK_BG/I-BEAM/I-BEAM_00006-view02.png"
images = [ob_image,c1_image,c2_image,c3_image,c4_image]

imagedir = "/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY"

cols = 4
rows = 5
cell_size = [400,360]

new_image = Image.new(size=(cols*cell_size[0]+40,rows*cell_size[1]+80),mode="RGB",color=(255,255,255))
draw = ImageDraw.Draw(new_image)
font = ImageFont.truetype("FreeMonoBold.ttf", 48)
font2 = ImageFont.truetype("FreeMonoBold.ttf", 32)
# draw.text((x, y),"Sample Text",(r,g,b))

for row in range(rows):
    rgb_image = cv2.imread(images[row])

    print(rgb_image.shape)
    black_channel = np.zeros((rgb_image.shape[0],rgb_image.shape[1],1), np.uint8)

    (tB, tG, tR) = cv2.split(rgb_image)
    split_r = cv2.merge([black_channel,black_channel,tR])
    split_g = cv2.merge([black_channel,tG,black_channel])
    split_b = cv2.merge([tB,black_channel,black_channel])
    temp_images = [split_r,split_g,split_b,rgb_image]
    for col in range(cols):
        #print(row,temp_images[row].shape)
        new_image.paste(Image.fromarray(cv2.cvtColor(temp_images[col], cv2.COLOR_BGR2RGB)),(cell_size[0]*col+50,cell_size[1]*row+50+40))
    
        draw.text((cell_size[0]*col+40+350, (row+1)*cell_size[1]-(cell_size[1]/2)-10+40),["+","+","=",""][col],(0,0,0),font=font)
    #draw.text((cell_size[0]*col+40+150, 800-24),"+",(0,0,0),font=font)
    #draw.text((cell_size[0]*col+40+150, 1200-24),"=",(0,0,0),font=font)
    #label = ["O\nV\nE\nR\nL\nA\nY","C\nO\nM\nP\nO\nS\nI\nT\nE\n1","C\nO\nM\nP\nO\nS\nI\nT\nE\n2","C\nO\nM\nP\nO\nS\nI\nT\nE\n3"][row]
    #label = ["S\nH\nA\nD\nE\nD","C\n1","C\n2","C\n3","C\n4"][row]
    label = ["SH","C1","C2","C3","C4"][row]
    label_size = font.getbbox(label)
    #print(label_size,font.getsize(label))
    draw.text((int(cell_size[0]*3)+370, 180+50+cell_size[1]*row-int(((label_size[3]-label_size[1])/2)*len(label)/2)),label,(0,0,0),font=font)
    if row < 4:
        draw.line([50,int(cell_size[1]*(row+1)+57),cell_size[0]*cols-10,int(cell_size[1]*(row+1)+57)], width = 2, fill=(0,0,0))


for i,t in enumerate([" R"," G"," B","RGB"]):
    draw.text((cell_size[0]*i+ int(cell_size[0]/2)-40, 20),t,(0,0,0),font=font)
#draw.text((int(cell_size[0]/2)-10, 20),"G",(0,0,0),font=font)
#draw.text((int(cell_size[0]/2)-10, 20),"B",(0,0,0),font=font)
#print(font.getsize("O\nV\nE\nR\nL\nA\nY"))
#draw.text((int(cell_size[0]*3), 120),"O\nV\nE\nR\nL\nA\nY",(0,0,0),font=font)
    

new_image.save("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/Figure_15.png")



# szum

image50 = Image.open("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/render_50samples.png")
image50dn = Image.open("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/render_50samplesDN.png")
image500 = Image.open("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/render_500samples.png")
image500dn = Image.open("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/render_500samplesDN.png")
images = [image50,image50dn,image500,image500dn]

#wbaa_image_crop = wbaa_image.crop((135, 135, 185, 185)).resize((240,240), Image.Resampling.)
crops = []
for temp_image in images:
    crops.append(temp_image.crop((230, 300, 290, 360)).resize((240,240), Image.Resampling.NEAREST))

cols = 4
rows = 1
cell_size = [700,700]

new_image = Image.new(size=(cols*cell_size[0],rows*cell_size[1]),mode="RGB",color=(255,255,255))
draw = ImageDraw.Draw(new_image)

for col in range(cols):
    new_image.paste(images[col],(cell_size[0]*col+50,50))
    new_image.paste(crops[col],(cell_size[0]*col+410,50))
    
    draw.rectangle([cell_size[0]*col+230+50,50+300,cell_size[0]*col+230+50+60,50+300+60], width = 2, outline=(255,0,255))
    draw.rectangle([cell_size[0]*col+410,50,cell_size[0]*col+410+240,50+240], width = 4, outline=(255,0,255))

    draw.line([cell_size[0]*col+230+50,50+300,cell_size[0]*col+410,50], width = 3, fill=(255,0,255))
    draw.line([cell_size[0]*col+230+50+60,50+300+60,cell_size[0]*col+410+240,50+240], width = 3, fill=(255,0,255))


new_image.save("/home/maciejm/PHD/PUBLIKACJA_02/OBRAZY/SZUM.png")
'''