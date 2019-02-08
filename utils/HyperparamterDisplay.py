from __future__ import print_function, absolute_import

def display(args):
    #  Display information of current training
    print('Data Set \t%s' % args.data)
    print('Network \t%s' % args.net)
    print('Loss Function \t%s' % args.loss)
    print('Embedded Dim \t%d' % args.dim)
    print('Image size \t%d' % args.width)
    print('Learn Rate  \t%.1e' % args.lr)
    print('Epochs  \t%05d' % args.epochs)
    print('Batch Size  \t%d' % args.batch_size)
    print('Num-Instance  \t%d' % args.num_instances)
    print('Log Path \t%s' % args.save_dir)
    # print('Number of Neighbour \t%d' % args.k)
    # print('Alpha \t %d' % args.alpha)

    print(40*'#')
