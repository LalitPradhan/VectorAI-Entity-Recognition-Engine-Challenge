import spacy
from nostril import nonsense
import sys
import pprint
import argparse
import pickle
import os

# Arguments to be passed
parser = argparse.ArgumentParser(description='Argumets for VectorAI Coding Challenge.')
parser.add_argument('--demo_flag', type=bool, default=False, help='set this to true to run a demo on pre written strings')
parser.add_argument('--similarity_threshold', type=float, default=0.7, help='set threshold between 0.0 to 1.0 for similarity matching')
parser.add_argument('--database_file_path', type=str, default='output.pkl', help='path to pickle file for historical data')
parser.add_argument('--use_database_flag', type=bool, default=False, help='set this to true to use historical data processed so far with the correct database_file_path')
parser.add_argument('--overwrite_flag', type=bool, default=False, help='set this to true to update the historical database with the new entries')
args = parser.parse_args()

# Load the large model of spacy
nlp = spacy.load("en_core_web_lg")

#data for demo file
stream1 = "MARKS   AND   SPENCERS   LTD"
stream2 = "Phone"
stream3 = "ICNAO02312"
stream4 = "LONDON,   GREAT   BRITAIN"
stream5 = "Sainsbury's Ltd."
stream6 = "INTEL   LLC"
stream7 = "M&S   CORPORATION   Limited"
stream8 = "LONDON,   ENGLAND"
stream9 = "Marks   and   Spencers   Ltd"
stream10= "M&S   Limited"
stream11= "NVIDIA Ireland"
stream12= "SLOUGH   SE12   2XY"
stream13= "33   TIMBER   YARD,  LONDON,   L1   8XY"
stream14= "44   CHINA   ROAD,   KOWLOON,   HONG  KONG"
stream15= "XYZ 13423 / ILD"
stream16= "ABC/ICL/20891NC"
stream17= "HARDWOOD TABLE"
stream18= "PLASTIC BOTTLE"
stream19= "ASIA"
stream20= "HONG KONG"
stream21= "LONDON, ENGLAND"
stream22= "Table Fan"
stream23= "Hammersmith, London"
stream24= "XYZ13423/ILD"
stream25= "Water Bottle"
stream26= "CIELING FAN"
streams = [stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8,stream9, 
stream10, stream11, stream12, stream13, stream14, stream15, stream16, stream17, stream18, stream19,
stream20, stream21, stream22, stream23, stream24, stream25, stream26]

#Classifier for the 5 categories
def classifier(stream_strings):
	"""Rule based classification for the input string

	Args:
	  str: A string to be processed

	Returns:
	  str: category/ class of the string
	"""

	# Augment the string with splits and elimination of spaces and changing to lower case
	stream_strings_org = stream_strings
	stream_strings = stream_strings.lower()
	stream_strings = ' '.join(stream_strings.split()).replace(',','')
	stream_strings_space = stream_strings.split(' ')
	stream_strings_mod = list(set(stream_strings_space + [stream_strings] + [stream_strings_org]))
	
	# Variables for rule based matching
	ents = []
	in_vocab = []
	poss = {}
	alphas = {}
	like_nums = {}


	for token in nlp(' '.join(stream_strings_org.split()).lower()): #Iterate through string after converting into a spacy doc
		poss[token.text] = str(token.pos_) #Part of Speech for the token
		alphas[token.text] = str(token.is_alpha) #Check if token is alphabetic
		like_nums[token.text] = str(token.like_num) #Check if token is numeric
		if nlp.vocab.strings[token.text] in nlp.vocab:
			in_vocab.append(token.text) #Check if token is present in vocabulary
	for stream_string in stream_strings_mod: # Check the entity labels
		doc = nlp(stream_string)
		for ent in doc.ents:
			if ent.label_ not in ents:
				ents.append(ent.label_)
	ents = list(set(ents))
	
	#Rule for serial number
	if (len(ents)==0 and len(in_vocab)==0) or nonsense(''.join(stream_strings_org.lower().split())*3)==True or ''.join(stream_strings_org.split()).isnumeric()==True:
		return 'serial_number'

	#Rule for physical good/ Product
	if (len(nlp(' '.join(stream_strings_org.split()))) == len(in_vocab)) and (len(ents)==0 or 'PRODUCT' in ents):
		return 'product'

	#Rule for Address
	if ('CARDINAL' in ents) or ('FAC' in ents):
		return 'address'

	#Rule for Geographic location
	if ('GPE' in ents or 'LOC' in ents) and ('ORG' not in ents):
		return 'location'

	#Rule for Company Name
	if 'ORG' in ents:
		return 'company_name'
	return 'product'


def demo():
	"""Run demo on a pre written list of strings
	This prints the prewritten data. Classifies and clusters the data into a dictionary and saves it in a pickle file.

	Args:
	  None: 

	Returns:
	  None: 
	"""

	#Initialize database
	if args.use_database_flag: #Check if historical database is to be used
		if os.path.isfile(args.database_file_path): #Check if file is already present
			with open(args.database_file_path,'rb') as fobj:
				database = pickle.load(fobj)
		else:
			database = {'company_name':  {}, 'address': {}, 'location': {}, 'product': {}, 'serial_number': {}}
	else:
		database = {'company_name':  {}, 'address': {}, 'location': {}, 'product': {}, 'serial_number': {}} # Initialize with empty database with the category names
	
	#Print Input Data
	pprint.pprint('Input Data:')
	pprint.pprint(streams)
	pprint.pprint('')
	
	for input_string in streams: #Iterate through the list of strings
		similarity_score = 0.0
		category = classifier(input_string) #Classify the input string
		if len(database[category]) == 0: #Check is any clusters for the particular category is present
			database[category]['cat0'] = []
			database[category]['cat0'].append(input_string)
			continue
		else:
			for category_id in database[category].keys(): #If there are entries in clusters then iterate through them to find similary measure
				for string_database in database[category][category_id]:
					if nlp(string_database).similarity(nlp(input_string)) >= similarity_score: #Update the closest cateogry to the highest similarity score
						similarity_score = nlp(string_database).similarity(nlp(input_string)) 
						nearest_category = category_id
		if similarity_score>=args.similarity_threshold: #If score above a threshold then add the string to closest cluster
			database[category][nearest_category].append(input_string)
		else:
			new_category = 'cat'+str(len(database[category])) #If score below a threshold then create a new cluster
			database[category][new_category] = []
			database[category][new_category].append(input_string)

	pprint.pprint('Output:')
	pprint.pprint(database)
	if args.overwrite_flag: #Save by updating the database.
		with open(args.database_file_path,'wb') as fobj:
			pickle.dump(database, fobj, pickle.HIGHEST_PROTOCOL)

def custom_input():
	"""Takes input from the user and classifies/ assigns to a particular set of cluster.

	Args:
	  None: 

	Returns:
	  None: 
	"""

	#Initialize database
	if args.use_database_flag: #Check if historical database is to be used
		if os.path.isfile(args.database_file_path): #Check if file is already present
			with open(args.database_file_path,'rb') as fobj:
				database = pickle.load(fobj)
		else:
			database = {'company_name':  {}, 'address': {}, 'location': {}, 'product': {}, 'serial_number': {}}
	else:
		database = {'company_name':  {}, 'address': {}, 'location': {}, 'product': {}, 'serial_number': {}} # Initialize with empty database with the category names


	while True: #keep taking inputs util one types quit
		input_string = input('Enter your string. Type quit to exit: ') 
		similarity_score = 0.0
		if input_string.lower() != 'quit':
			category = classifier(input_string) #Classify the input string
			if len(database[category]) == 0: #Check is any clusters for the particular category is present
				database[category]['cat0'] = []
				database[category]['cat0'].append(input_string)
				continue
			else:
				for category_id in database[category].keys():  #If there are entries in clusters then iterate through them to find similary measure
					for string_database in database[category][category_id]:
						if nlp(string_database).similarity(nlp(input_string)) >= similarity_score: #Update the closest cateogry to the highest similarity score
							similarity_score = nlp(string_database).similarity(nlp(input_string))
							nearest_category = category_id
			if similarity_score>=args.similarity_threshold: #If score above a threshold then add the string to closest cluster
				database[category][nearest_category].append(input_string)
			else:
				new_category = 'cat'+str(len(database[category])) #If score below a threshold then create a new cluster
				database[category][new_category] = []
				database[category][new_category].append(input_string)
		else:
			pprint.pprint(database) #Print the database
			if args.overwrite_flag: #Update the existing database with new entries and save
				with open(args.database_file_path,'wb') as fobj:
					pickle.dump(database, fobj, pickle.HIGHEST_PROTOCOL)
			sys.exit() #Exit code


	

if __name__ == '__main__':

	if args.demo_flag:
		demo() #Run the demo
	else:
		custom_input() #Run manual entries
