setup:
	mkdir -p flowers && cd flowers
	curl https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz -o flowers/flower_data.tar.gz
	tar -zxf flowers/flower_data.tar.gz -C flowers/

make notebook:
	jupyter notebook
