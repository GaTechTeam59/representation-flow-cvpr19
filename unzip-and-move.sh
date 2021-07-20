mkdir hmdb51_sta_unzipped
cd hmdb51_sta
for file in *.rar; do
    unrar x $file ../hmdb51_sta_unzipped
done

cd ../hmdb51_sta_unzipped
for folder in *; do
    cd $folder
    for file in *.tform.mat; do
        rm $file
    done
    cd ..
done