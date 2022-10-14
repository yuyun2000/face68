import os

#写入CatA、B、C 因为重复的太多了，每一个人只取30张
list = os.listdir('./data/label/300VW-3D')
file = open('labellistless.txt', mode='a+')
for i in range(len(list)):
    list2 = os.listdir('./data/label/300VW-3D/%s'%list[i])
    for j in range(len(list2)):
        list3 = os.listdir('./data/label/300VW-3D/%s/%s'%(list[i],list2[j]))
        num = 0
        for k in range(len(list3)):
            num +=1
            if num == 30:
                break
            file.write('./data/label/300VW-3D/%s/%s/%s \n'%(list[i],list2[j],list3[k]))
file.close()


file = open('labellistless.txt', mode='a+')
list = os.listdir('./data/label/AFLW2000-3D-Reannotated')
for i in range(len(list)):
    file.write('./data/label/AFLW2000-3D-Reannotated/%s \n' % (list[i]))
file.close()

file = open('labellistless.txt', mode='a+')
list = os.listdir('./data/label/Menpo-3D')
for i in range(len(list)):
    file.write('./data/label/Menpo-3D/%s \n' % (list[i]))
file.close()

file = open('labellistless.txt', mode='a+')
list = os.listdir('./data/label/new_dataset')
for i in range(len(list)):
    file.write('./data/label/new_dataset/%s \n' % (list[i]))
file.close()