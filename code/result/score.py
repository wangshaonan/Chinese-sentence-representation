import sys

maxscore = 0
maxline = ''
lastline = ''
for line in open(sys.argv[1]):
    if not line[0] == '0':
	continue
    line0 = line.strip().split()
    lastline = line
    if line0[0] > maxscore:
	maxscore = line0[0]
        maxline = line

#print 'lastline:' + lastline
print 'maxline:' + maxline
