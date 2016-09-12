from bs4 import BeautifulSoup

soup = BeautifulSoup(open("testData.xml"), "lxml")

#tag_department = soup.department  #gets the department but only the first one
tag_us_patent_grant = soup("us-patent-grant")
print("Total number of Patents " + str(len(tag_us_patent_grant)))

for patent in tag_us_patent_grant:
	
	tag_doc_number = patent.find({"doc-number"}).string    #finds first doc-number and none after
	print("patent num: %s" % tag_doc_number)

	'''
		Classifications
	'''

	tag_classification_national = patent.find({"classification-national"})
	#regEx to split up main and sub
	tag_main_classification = tag_classification_national("main-classification")  #some have a space in the xml file instead of a "/" that shows up on the patent files. the main class is before the space or "/" and subclass is after
	for each in tag_main_classification:
		print("US main cls: %s" % each.get_text())


	
	'''
		Art Department
	'''
	tag_department = patent.find("department").string
	print tag_department

	'''
		Specification
						looking for: <description id="description">
							Headings are irrelevant: <heading id="h-0002" level="1">BACKGROUND OF THE INVENTION</heading>
							Just need text that is different and it's all in: <p id="p-0003" num="0002">
						clean up with tensorflow
	'''
	tag_description = patent.find("description")
	tag_description_p = tag_description("p")
	for p in tag_description_p:
		tag_description_p_text = p.get_text()
	#	print tag_description_p_text
	'''
		Claims
						full category: <claims id="claims">
							individual claim: <claim id="CLM-00001" num="00001">
								claim text for each new line break or new line in claim: <claim-text>
	'''
#**************** below This doesn't print for the last one!!!!!	
	claims = []
	tag_claims = patent("claim") #finds the section of "claims" which contains the claim info
	for claim in tag_claims:
		claims.append(claim.get_text())
		print len(claims)
		print("this works") #this doesn't print for the last patent

#############not printing claims for last claim in test data##########

	#tag_claims_claim = tag_claims("claim-text") #finds all tags for "claim" which is separate for each individual claim
	##for claim in tag_claims:
		#tag_claims_claim_text = claim("claim-text")
	##	print claim

		#print tag_claims_claim_text
	#tag_classification_ipc = patent.find({"classification-ipc"})
	#tag_main_classification_ipc = tag_classification_ipc("main-classification").string
	#print("IPC main cls: %s" % tag_main_classification_ipc)

##	tag_department = patent.find("department").string

##	
	
	#tag_main_classification = tag_classification_national("main-classification")
	#print tag_main_classification

	#tag_further_classification = tag_classification_national("further-classification")
	#print tag_further_classification

	#tag_field_of_search = patent("field-of-search")
	#for each in tag_field_of_search:
	#	tag_field_of_search_main_classification = each.find("main-classification").string # using .find I can then use .string but am I getting all of them? NO!
	#	print("Field of Search Classes: " + tag_field_of_search_main_classification.string)
	#for each in tag_classification:
	#	tag_main_classes = tag_classification("main-classification")
	#	print tag_main_classes
	#for each in tag_field_of_search:
	#	print each.string
##	print("patent number: " + tag_doc_number + " |  Art Dept: %s" % tag_department)

	#print type(patent.find("department").string)

	#print tag_doc_number + tag_department	#prints all of doc then all of dept		#prints [<department>2132</department>, <department>2132</department>]
