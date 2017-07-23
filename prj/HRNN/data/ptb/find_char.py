file = open('train.txt', 'r')
letters = []
for line in file:
    for character in line:
	    if character not in letters:
		    letters.append(character)
file.close()
print(letters)
print(len(letters))
