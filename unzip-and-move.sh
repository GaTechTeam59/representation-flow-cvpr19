mkdir hmdb51_org_unzipped
cd hmdb51_org
for file in *.rar; do
    unrar x $file ../hmdb51_org_unzipped
done

cd ../hmdb51_org_unzipped
for folder in *; do
    cd $folder
    for file in *.tform.mat; do
        rm $file
    done
    cd ..
done