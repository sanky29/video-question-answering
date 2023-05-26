import matplotlib.pyplot as plt 
import numpy as np
import pdb
from argparse import ArgumentParser

def get_parser():
    
    #the parser for the arguments
    parser = ArgumentParser(
                        prog = 'python plot.py',
                        description = 'this is utility to plot the data quickly',
                        epilog = 'thank you!!')

    parser.add_argument('--title', nargs='+', required=False,  default = ['plot', 'title'], 
                        help='title of plot')

    parser.add_argument('--files', nargs='+', required=False,  default = [], 
                        help='name of files having data')


    parser.add_argument('--legends', nargs='+', required=False,  default = [], 
                        help='legends for the plots')

    parser.add_argument('--xlabel',  default = 'epoch', help='x label for plot')

    parser.add_argument('--ylabel',  default = 'crossentropy loss', help='y label for plot')
    
    parser.add_argument('--column',  nargs='+', help='column in the file', default=[0])
    parser.add_argument('--file_name',  default = 'plot.png' , help='name of file of plot')
    
    return parser


def read_from_file(file_name, column):
    data = []
    with open(file_name, "r") as f:
        
        #ignore the first line
        lines = f.readlines()
        lines = lines[1:]

        #the line of file is of type
        #value1,value2,\n 
        data = [float(line.split(',')[column]) for line in lines]
        return data

if __name__ == '__main__':

    #parse the arguments
    parser = get_parser()
    args = parser.parse_args()

    #filter columns
    if(len(args.column) == 1):
        args.column = [args.column[0] for i in args.files]
    args.column = [int(i) for i in args.column]
    
    #read the datas
    datas = []
    for (i,file) in enumerate(args.files):
        datas.append(read_from_file(file, args.column[i]))
    
    #find the min len of data
    n = min([len(data) for data in datas])

    #set the x range
    x = np.arange(n)

    #now just plot
    for data in datas:
        plt.plot(x, data)
    
    #add legends
    if(len(args.legends) > 0):
        plt.legend(args.legends)
    
    #title and labels
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(' '.join(args.title))

    #save the plot
    plt.savefig(args.file_name)