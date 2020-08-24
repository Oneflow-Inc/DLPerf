echo "gzip train/demo_wiki_train.gz ..."
gzip -d train/demo_wiki_train.gz
echo "Done!"

for i in {1..50}                  
do
  cat train/demo_wiki_train >> train/demo_wiki_train_50        
  echo "Copy $i times"
  let i+=1
done

rm train/demo_wiki_train
echo "gzip demo_wiki_train_50 to demo_wiki_train_50.gz ..."
gzip  train/demo_wiki_train_50
echo "Success!"
