

# Introduction to Linux

## Preparation

1. Boot from a usb stick (or live cd), we suggest to use  [Ubuntu gnome](http://ubuntugnome.org/) distribution, or another ubuntu derivative.

2. (Optional) Configure keyboard layout and software repository
   Go to the the *Activities* menu (top left corner, or *start* key):
      -  Go to settings, then keyboard. Set the layout for latin america
      -  Go to software and updates, and select the server for Colombia
3. (Optional) Instead of booting from a live Cd. Create a partition in your pc's hard drive and install the linux distribution of your choice, the installed Os should perform better than the live cd.

## Introduction to Linux

1. Linux Distributions

   Linux is free software, it allows to do all sort of things with it. The main component in linux is the kernel, which is the part of the operating system that interfaces with the hardware. Applications run on top of it. 
   Distributions pack together the kernel with several applications in order to provide a complete operating system. There are hundreds of linux distributions available. In
   this lab we will be using Ubuntu as it is one of the largest, better supported, and user friendly distributions.


2. The graphical interface

   Most linux distributions include a graphical interface. There are several of these available for any taste.
   (http://www.howtogeek.com/163154/linux-users-have-a-choice-8-linux-desktop-environments/).
   Most activities can be accomplished from the interface, but the terminal is where the real power lies.

### Playing around with the file system and the terminal
The file system through the terminal
   Like any other component of the Os, the file system can be accessed from the command line. Here are some basic commands to navigate through the file system

   -  ``ls``: List contents of current directory
   - ``pwd``: Get the path  of current directory
   - ``cd``: Change Directory
   - ``cat``: Print contents of a file (also useful to concatenate files)
   - ``mv``: Move a file
   - ``cp``: Copy a file
   - ``rm``: Remove a file
   - ``touch``: Create a file, or update its timestamp
   - ``echo``: Print something to standard output
   - ``nano``: Handy command line file editor
   - ``find``: Find files and perform actions on it
   - ``which``: Find the location of a binary
   - ``wget``: Download a resource (identified by its url) from internet 

Some special directories are:
   - ``.`` (dot) : The current directory
   -  ``..`` (two dots) : The parent of the current directory
   -  ``/`` (slash): The root of the file system
   -  ``~`` (tilde) :  Home directory
      
Using these commands, take some time to explore the ubuntu filesystem, get to know the location of your user directory, and its default contents. 
   
To get more information about a command call it with the ``--help`` flag, or call ``man <command>`` for a more detailed description of it, for example ``man find`` or just search in google.


## Input/Output Redirections
Programs can work together in the linux environment, we just have to properly 'link' their outputs and their expected inputs. Here are some simple examples:

1. Find the ```passwd```file, and redirect its contents error log to the 'Black Hole'
   >  ``find / -name passwd  2> /dev/null``

   The `` 2>`` operator redirects the error output to ``/dev/null``. This is a special file that acts as a sink, anything sent to it will disappear. Other useful I/O redirection operations are
      -  `` > `` : Redirect standard output to a file
      -  `` | `` : Redirect standard output to standard input of another program
      -  `` 2> ``: Redirect error output to a file
      -  `` < `` : Send contents of a file to standard input
      -  `` 2>&1``: Send error output to the same place as standard output

2. To modify the content display of a file we can use the following command. It sends the content of the file to the ``tr`` command, which can be configured to format columns to tabs.

   ```bash
   cat milonga.txt | tr '\n' ' '
   ```
   
## SSH - Server Connection

1. The ssh command lets us connect to a remote machine identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER (**vision** in our case). The second command allows us to copy files between systems (you will get the actual login information in class).

   ```bash
   
   #connect
   ssh USER@SERVER
   ```

2. The scp command allows us to copy files form a remote server identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER. Following the SERVER information, we add ':' and write the full path of the file we want to copy, finally we add the local path where the file will be copied (remember '.' is the current directory). If we want to copy a directory we add the -r option. for example:

   ```bash
   #copy 
   scp USER@SERVER:~/data/sipi_images .
   
   scp -r USER@SERVER:/data/sipi_images .
   ```
   
   Notice how the first command will fail without the -r option

See [here](ssh.md) for different types of SSH connection with respect to your OS.

## File Ownership and permissions   

   Use ``ls -l`` to see a detailed list of files, this includes permissions and ownership
   Permissions are displayed as 9 letters, for example the following line means that the directory (we know it is a directory because of the first *d*) *images*
   belongs to user *vision* and group *vision*. Its owner can read (r), write (w) and access it (x), users in the group can only read and access the directory, while other users can't do anything. For files the x means execute. 
   ```bash
   drwxr-x--- 2 vision vision 4096 ene 25 18:45 images
   ```
   
   -  ``chmod`` change access permissions of a file (you must have write access)
   -  ``chown`` change the owner of a file
   
## Sample Exercise: Image database

1. Create a folder with your Uniandes username. (If you don't have Linux in your personal computer)

2. Copy *sipi_images* folder to your personal folder. (If you don't have Linux in your personal computer)

3.  Decompress the images (use ``tar``, check the man) inside *sipi_images* folder. 

4.  Use  ``imagemagick`` to find all *grayscale* images. We first need to install the *imagemagick* package by typing

    ```bash
    sudo apt-get install imagemagick
    ```
    
    Sudo is a special command that lets us perform the next command as the system administrator
    (super user). In general it is not recommended to work as a super user, it should only be used 
    when it is necessary. This provides additional protection for the system.
    
    ```bash
    find . -name "*.tiff" -exec identify {} \; | grep -i gray | wc -l
    ```
    
3.  Create a script to copy all *color* images to a different folder
    Lines that start with # are comments
       
      ```bash
      #!/bin/bash
      
      # go to Home directory
      cd ~ # or just cd

      # remove the folder created by a previous run from the script
      rm -rf color_images

      # create output directory
      mkdir color_images

      # find all files whose name end in .tif
      images=$(find sipi_images -name *.tiff)
      
      #iterate over them
      for im in ${images[*]}
      do
         # check if the output from identify contains the word "gray"
         identify $im | grep -q -i gray
         
         # $? gives the exit code of the last command, in this case grep, it will be zero if a match was found
         if [ $? -eq 0 ]
         then
            echo $im is gray
         else
            echo $im is color
            cp $im color_images
         fi
      done
      
      ```
      -  save it for example as ``find_color_images.sh``
      -  make executable ``chmod u+x`` (This means add Execute permission for the user)
      -  run ``./find_duplicates.sh`` (The dot is necessary to run a program in the current directory)
      

## Your turn

1. What is the ``grep``command?
The "grep" command allows to identify text patterns in a specified file. Also, "grep" can search over subdirectories and through 
several files.In addition "grep" can search for patterns in more complex structures e.g. looking for "dog", "grep" can identify 
"god","odg","gdo" and so on. For this, it can be specified the type of file in which "grep" must do the search, e.g. " .txt or .html".
Source: https://www.computerhope.com/unix/ugrep.htm
2. What is the meaning of ``#!/bin/python`` at the start of scripts?
the line "#!/bin/python" specifies the type of executable that the file is so, in this case, it can be open as a python executable.
Source: https://martin-thoma.com/what-does-usrbinpython-mean/
3. Download using ``wget`` the [*bsds500*](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) image segmentation database, and decompress it using ``tar`` (keep it in you hard drive, we will come back over this data in a few weeks).
 
4. What is the disk size of the uncompressed dataset, How many images are in the directory 'BSR/BSDS500/data/images'?
The disk size of the uncompressed dataset is 69.105MB. There are 201 test images, 201 train images and 101 val images. 
5. What are all the different resolutions? What is their format? Tip: use ``awk``, ``sort``, ``uniq`` 
Using the command "file*"; In train,test and val some of the images have a resolution of 481x321 and others a resolution of 321x481.
All of the images are in JPEG format. This can also be done using the command "identify*"  with the imagemagick package installed.
Source: https://www.computerhope.com/unix/ufile.htm
6. How many of them are in *landscape* orientation (opposed to *portrait*)? Tip: use ``awk`` and ``cut``
! [] (data:image/jpeg;base64,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)

7. Crop all images to make them square (256x256) and save them in a different folder. Tip: do not forget about  [imagemagick](http://www.imagemagick.org/script/index.php).


# Report

For every question write a detailed description of all the commands/scripts you used to complete them. DO NOT use a graphical interface to complete any of the tasks. Use screenshots to support your findings if you want to. 

Feel free to search for help on the internet, but ALWAYS report any external source you used.

Notice some of the questions actually require you to connect to the course server, the login instructions and credentials will be provided on the first session. 

## Deadline

We will be delivering every lab through the [github](https://github.com) tool (Silly link isn't it?). According to our schedule we will complete that tutorial on the second week, therefore the deadline for this lab will be specially long **February 7 11:59 pm, (it is the same as the second lab)** 

### More information on

http://www.ee.surrey.ac.uk/Teaching/Unix/ 




