import re

patList = []

##open file
f = open('testData.txt', 'r')  #open the file

## create string from text in file
text = f.read()
print type(text)

## find each patent in file and return the text as a string to a list (one patent string per space in list)
patents = re.split('</us-patent-grant>', text)
print len(patents)  # verify if got all patents. last space should be empty since split returns tuple and split string is at end

for patent in patents:
	patdict = {}
	## find patent number/identifier (find--->file="US06979701-20051227.XML") WORKING
	patent_number_row = re.findall ( '<us-patent-grant(.*?)id="us-patent-grant"', patent, re.DOTALL)
	patent_number_row = str(patent_number_row) #convert to string for string input on next line
	patent_number = re.findall ( 'file="(.*?)-', patent_number_row, re.DOTALL)
#	print ("Patent number: %s " % str(patent_number))
	patdict['patnum'] = patent_number


	'''
	</us-term-of-grant>
	<classification-ipc>
	<edition>7</edition>
	<main-classification>C08G018/06</main-classification>
	</classification-ipc>
	<classification-national>
	<country>US</country>
	<main-classification>521170</main-classification>
	<further-classification>521128</further-classification>
	<further-classification>521130</further-classification>
	<further-classification>521163</further-classification>
	<further-classification>521174</further-classification>
	</classification-national>
	<invention-title
	'''
	## find US classifications  WORKING
	classification_row = re.findall ( '</us-term-of-grant>(.*?)<invention-title', patent, re.DOTALL)
	classification_row = str(classification_row)
	us_class_row = re.findall ( '<country>US</country>(.*?)</classification-national>', classification_row, re.DOTALL)
	us_class_row = str(us_class_row)
	main_classification = re.findall ( '<main-classification>(.*?)<', us_class_row, re.DOTALL)
#	print ("US main class: %s " % str(main_classification))
	patdict['usMainClass'] = main_classification

	further_classification = re.findall ( '<further-classification>(.*?)</further-classification>', us_class_row, re.DOTALL)
#	print ("Further US classes: %s " % str(further_classification))
	patdict['usFurtherClasses'] = further_classification

	## find IPC classifications WORKING
	ipc_class_row = re.findall ( '<classification-ipc>(.*?)</classification-ipc>', classification_row, re.DOTALL)
	ipc_class_row = str(ipc_class_row)
	full_ipc = re.findall ( '<main-classification>(.*?)<', ipc_class_row, re.DOTALL)
#	print ("IPC class: %s " % str(full_ipc))
	patdict['classIPC'] = full_ipc

	## find search classifications WORKING
	'''
	<field-of-search>
	<classification-national>
	<country>US</country>
	<main-classification>D 1106</main-classification>
	</classification-national>
	<classification-national>
	<country>US</country>
	<main-classification>D 1126-127</main-classification>
	<additional-info>unstructured</additional-info>
	</classification-national>
	<classification-national>
	<country>US</country>
	<main-classification>426104</main-classification>
	</classification-national>
	<classification-national>
	<country>US</country>
	<main-classification>426516</main-classification>
	</classification-national>
	<classification-national>
	<country>US</country>
	<main-classification>D21386</main-classification>
	</classification-national>
	<classification-national>
	<country>US</country>
	<main-classification>D21402</main-classification>
	</classification-national>
	<classification-national>
	<country>US</country>
	<main-classification>D11 70</main-classification>
	</classification-national>
	</field-of-search>
	'''
	search_classification_row = re.findall ( '<field-of-search>(.*?)</field-of-search>', patent, re.DOTALL)
	search_classification_row = str(search_classification_row)
	search_main_class_row = re.findall ( '<main-classification>(.*?)</main-classification>', search_classification_row, re.DOTALL)
#	print ("Field of search: %s " % str(search_main_class_row))
	patdict['searchClass'] = search_main_class_row

	## find art unit
	'''
	<examiners>
	<primary-examiner>
	<last-name>Burgess</last-name>
	<first-name>Pamela</first-name>
	<department>2911</department>
	</primary-examiner>
	</examiners>
	'''
	art_unit_row = re.findall ( '<examiners>(.*?)</examiners>', patent, re.DOTALL)
	art_unit_row = str(art_unit_row)
	art_unit = re.findall ( '<department>(.*?)</department>', art_unit_row, re.DOTALL)
#	print ("Art Unit: %s " % str(art_unit))
	patdict['artUnit'] = art_unit

	## find each description WORKING
	description = re.findall ( '<description id="description">(.*?)</description>', patent, re.DOTALL)
#	print description  #prints the last blank one from patents
	patdict['description'] = description

	## find each claim set WORKING
	claims = re.findall ( '<claims id="claims">(.*?)</claims>', patent, re.DOTALL)
#	print claims
	patdict['claims'] = claims

	patList.append(patdict)
#############################
	# write to csv file #

	# https://docs.python.org/2/library/csv.html
#############################	
	


f.close()  #close the file when finished

print len(patList)

