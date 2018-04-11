import sys
import datetime
import numpy as np
import scipy
import pandas

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def pngExample():
	print "Main..."

	t = np.arange(0.0, 2.0, 0.01)
	s = 1 + np.sin(2*np.pi*t)
	plt.plot(t, s)

	plt.xlabel('time (s)')
	plt.ylabel('voltage (mV)')
	plt.title('About as simple as it gets, folks')
	plt.grid(True)
	plt.savefig("test.png")
	plt.show()

	fignum = np.random.randint(0,sys.maxint)
      	plt.figure(fignum)

def pdfExample():
	with PdfPages('multipage_pdf.pdf') as pdf:
	    plt.figure(figsize=(3, 3))
	    plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
	    plt.title('Page One')
	    pdf.savefig()  # saves the current figure into a pdf page
	    plt.close()

	    plt.rc('text', usetex=True)
	    plt.figure(figsize=(8, 6))
	    x = np.arange(0, 5, 0.1)
	    plt.plot(x, np.sin(x), 'b-')
	    plt.title('Page Two')
	    pdf.attach_note("plot of sin(x)")  # you can add a pdf note to
					       # attach metadata to a page
	    pdf.savefig()
	    plt.close()

	    plt.rc('text', usetex=False)
	    fig = plt.figure(figsize=(4, 5))
	    plt.plot(x, x*x, 'ko')
	    plt.title('Page Three')
	    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
	    plt.close()

	    # We can also set the file's metadata via the PdfPages object:
	    d = pdf.infodict()
	    d['Title'] = 'Multipage PDF Example'
	    d['Author'] = u'Jouni K. Sepp\xe4nen'
	    d['Subject'] = 'How to create a multipage pdf file and set its metadata'
	    d['Keywords'] = 'PdfPages multipage keywords author title subject'
	    d['CreationDate'] = datetime.datetime(2009, 11, 13)
	    d['ModDate'] = datetime.datetime.today()


if __name__ == "__main__":
	pdfExample()
	pngExample()
