<h1><div align="center">Asynchronous Streaming, and Visual Profiling with CUDA C/C++</div></h1>

![CUDA](./images/CUDA_Logo.jpg)

The CUDA toolkit ships with the **Nsight Systems**, a powerful GUI application to support the development of accelerated CUDA applications. Nsight Systems generates a graphical timeline of an accelerated application, with detailed information about CUDA API calls, kernel execution, memory activity, and the use of **CUDA streams**.

In this lab, you will be using the Nsight Systems timeline to guide you in optimizing accelerated applications. Additionally, you will learn some intermediate CUDA programming techniques to support your work: **unmanaged memory allocation and migration**; **pinning**, or **page-locking** host memory; and **non-default concurrent CUDA streams**.

At the end of this lab, you will be presented with an assessment, to accelerate and optimize a simple n-body particle simulator, which will allow you to demonstrate the skills you have developed during this course. Those of you who are able to accelerate the simulator while maintaining its correctness, will be granted a certification as proof of your competency.

---
## Prerequisites

To get the most out of this lab you should already be able to:

- Write, compile, and run C/C++ programs that both call CPU functions and launch GPU kernels.
- Control parallel thread hierarchy using execution configuration.
- Refactor serial loops to execute their iterations in parallel on a GPU.
- Allocate and free CUDA Unified Memory.
- Understand the behavior of Unified Memory with regard to page faulting and data migrations.
- Use asynchronous memory prefetching to reduce page faults and data migrations.

## Objectives

By the time you complete this lab you will be able to:

- Use **Nsight Systems** to visually profile the timeline of GPU-accelerated CUDA applications.
- Use Nsight Systems to identify, and exploit, optimization opportunities in GPU-accelerated CUDA applications.
- Utilize CUDA streams for concurrent kernel execution in accelerated applications.
- (**Optional Advanced Content**) Use manual device memory allocation, including allocating pinned memory, in order to asynchronously transfer data in concurrent CUDA streams.

---
## Running Nsight Systems

For this interactive lab environment, we have set up a remote desktop you can access from your browser, where you will be able to launch and use Nsight Systems.

You will begin by creating a report file for an already-existing vector addition program, after which you will be walked through a series of steps to open this report file in Nsight Systems, and to make the visual experience nice.

### Generate Report File

[`01-vector-add.cu`](../edit/01-vector-add/01-vector-add.cu) (<-------- click on these links to source files to edit them in the browser) contains a working, accelerated, vector addition application. Use the code execution cell directly below (you can execute it, and any of the code execution cells in this lab by `CTRL` + clicking it) to compile and run it. You should see a message printed that indicates it was successful.


```python
!nvcc -o vector-add-no-prefetch 01-vector-add/01-vector-add.cu -run
```

    Success! All values calculated correctly.


Next, use `nsys profile --stats=true` to create a report file that you will be able to open in the Nsight Systems visual profiler. Here we use the `-o` flag to give the report file a memorable name:


```python
!nsys profile --stats=true -o vector-add-no-prefetch-report ./vector-add-no-prefetch
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Success! All values calculated correctly.
    Processing events...
    Saving temporary "/tmp/nsys-report-6a5a-6eb4-e9e5-8890.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-6a5a-6eb4-e9e5-8890.qdrep"
    Exporting 10235 events: [=================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-6a5a-6eb4-e9e5-8890.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls    Average     Minimum    Maximum           Name         
     -------  ---------------  ---------  -----------  ---------  ---------  ---------------------
        61.9        246083765          3   82027921.7      17255  246026906  cudaMallocManaged    
        32.5        129021842          1  129021842.0  129021842  129021842  cudaDeviceSynchronize
         5.6         22384952          3    7461650.7    6701513    8922765  cudaFree             
         0.0            45634          1      45634.0      45634      45634  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances    Average     Minimum    Maximum                      Name                    
     -------  ---------------  ---------  -----------  ---------  ---------  -------------------------------------------
       100.0        129012278          1  129012278.0  129012278  129012278  addVectorsInto(float*, float*, float*, int)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
     -------  ---------------  ----------  -------  -------  -------  ---------------------------------
        78.8         78911017        8314   9491.3     2143   128189  [CUDA Unified Memory memcpy HtoD]
        21.2         21192428         768  27594.3     1567   159742  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total     Operations  Average  Minimum  Maximum               Operation            
     ----------  ----------  -------  -------  --------  ---------------------------------
     393216.000        8314   47.296    4.000   764.000  [CUDA Unified Memory memcpy HtoD]
     131072.000         768  170.667    4.000  1020.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum        Name     
     -------  ---------------  ---------  ----------  -------  ---------  --------------
        83.3       1330490805         72  18479039.0    32582  100127978  poll          
         8.8        140558156         63   2231081.8    15224   20538800  sem_timedwait 
         6.3         99891506        676    147768.5     1070   18788038  ioctl         
         1.6         24916487         94    265069.0     1407    8868714  mmap          
         0.1          1469485         82     17920.5     4713      28255  open64        
         0.0           214257          3     71419.0    69421      74578  fgets         
         0.0           154636          4     38659.0    35417      46043  pthread_create
         0.0           117941         25      4717.6     1563      25048  fopen         
         0.0            77800         11      7072.7     4800      11352  write         
         0.0            41449         11      3768.1     1443       5929  munmap        
         0.0            28638          5      5727.6     3301       8493  open          
         0.0            26997         18      1499.8     1113       3778  fclose        
         0.0            22073          6      3678.8     1091       9657  fgetc         
         0.0            21042         13      1618.6     1047       2401  read          
         0.0            14895          3      4965.0     1663       7476  fread         
         0.0            10453          7      1493.3     1002       3690  fcntl         
         0.0             9649          2      4824.5     4477       5172  socket        
         0.0             7203          1      7203.0     7203       7203  pipe2         
         0.0             6243          1      6243.0     6243       6243  connect       
         0.0             2535          1      2535.0     2535       2535  bind          
         0.0             1461          1      1461.0     1461       1461  listen        
    
    Report file moved to "/dli/task/vector-add-no-prefetch-report.qdrep"
    Report file moved to "/dli/task/vector-add-no-prefetch-report.sqlite"
    


### Open the Remote Desktop

Run the next cell to generate a link to the remote desktop. Then, read the instructions that follow in the notebook.


```python
%%js
var port = ((window.location.port == 80) ? "" : (":"+window.location.port));
var url = 'http://' + window.location.hostname + port + '/nsight/vnc.html?resize=scale';
let a = document.createElement('a');
a.setAttribute('href', url)
a.setAttribute('target', '_blank')
a.innerText = 'Click to open remote desktop'
element.append(a);
```


    <IPython.core.display.Javascript object>


After clicking the _Connect_ button you will be asked for a password, which is `nvidia`.

### Open Nsight Systems

To open Nsight Systems, double-click the "NVIDIA Nsight Systems" icon on the remote desktop.

![open nsight](images/open-nsight-sys.png)

### Enable Usage Reporting

When prompted, click "Yes" to enable usage reporting:

![enable usage](images/enable_usage.png)

### Select GPU Rows on Top

When prompted, select _GPU Rows on Top_ and then click _Okay_.

![gpu)_rows_on_top](images/gpu_on_top.png)

### Open the Report File

Open this report file by visiting _File_ -> _Open_ from the Nsight Systems menu and select `vector-add-no-prefetch-report.qdrep`:

![open-report](images/open-report.png)

### Ignore Warnings/Errors

You can close and ignore any warnings or errors you see, which are just a result of our particular remote desktop environment:

![ignore errors](images/ignore-error.png)

### Make More Room for the Timelines

To make your experience nicer, full-screen the profiler, close the _Project Explorer_ and hide the *Events View*:

![make nice](images/make-nice.png)

Your screen should now look like this:

![now nice](images/now-nice.png)

### Expand the CUDA Unified Memory Timelines

Next, expand the _CUDA_ -> _Unified memory_ and _Context_ timelines, and close the _Threads_ timelines:

![open memory](images/open-memory.png)

### Observe Many Memory Transfers

From a glance you can see that your application is taking about 1 second to run, and that also, during the time when the `addVectorsInto` kernel is running, that there is a lot of UM memory activity:

![memory and kernel](images/memory-and-kernel.png)

Zoom into the memory timelines to see more clearly all the small memory transfers being caused by the on-demand memory page faults. A couple tips:

1. You can zoom in and out at any point of the timeline by holding `CTRL` while scrolling your mouse/trackpad
2. You can zoom into any section by click + dragging a rectangle around it, and then selecting _Zoom in_

Here's an example of zooming in to see the many small memory transfers:

![many transfers](images/many-transfers.png)

---
## Comparing Code Refactors Iteratively with Nsight Systems

Now that you have Nsight Systems up and running and are comfortable moving around the timelines, you will be profiling a series of programs that were iteratively improved using techniques already familiar to you. Each time you profile, information in the timeline will give information supporting how you should next modify your code. Doing this will further increase your understanding of how various CUDA programming techniques affect application performance.

### Exercise: Compare the Timelines of Prefetching vs. Non-Prefetching

[`01-vector-add-prefetch-solution.cu`](../edit/01-vector-add/solutions/01-vector-add-prefetch-solution.cu) refactors the vector addition application from above so that the 3 vectors needed by its `addVectorsInto` kernel are asynchronously prefetched to the active GPU device prior to launching the kernel (using [`cudaMemPrefetchAsync`](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8dc9199943d421bc8bc7f473df12e42)). Open the source code and identify where in the application these changes were made.

After reviewing the changes, compile and run the refactored application using the code execution cell directly below. You should see its success message printed.


```python
!nvcc -o vector-add-prefetch 01-vector-add/solutions/01-vector-add-prefetch-solution.cu -run
```

    Success! All values calculated correctly.


Now create a report file for this version of the application:


```python
!nsys profile --stats=true -o vector-add-prefetch-report ./vector-add-prefetch
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Success! All values calculated correctly.
    Processing events...
    Saving temporary "/tmp/nsys-report-02b7-473d-4ff0-86b9.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-02b7-473d-4ff0-86b9.qdrep"
    Exporting 2126 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-02b7-473d-4ff0-86b9.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum    Maximum           Name         
     -------  ---------------  ---------  ----------  --------  ---------  ---------------------
        73.9        260466014          3  86822004.7     18188  260391147  cudaMallocManaged    
        16.7         58973167          1  58973167.0  58973167   58973167  cudaDeviceSynchronize
         6.4         22708163          3   7569387.7   6837370    8952196  cudaFree             
         2.9         10368727          3   3456242.3      6106   10236373  cudaMemPrefetchAsync 
         0.0            33007          1     33007.0     33007      33007  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances   Average   Minimum  Maximum                     Name                    
     -------  ---------------  ---------  ---------  -------  -------  -------------------------------------------
       100.0          1697611          1  1697611.0  1697611  1697611  addVectorsInto(float*, float*, float*, int)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average   Minimum  Maximum              Operation            
     -------  ---------------  ----------  --------  -------  -------  ---------------------------------
        75.7         65658107         192  341969.3   339868   344220  [CUDA Unified Memory memcpy HtoD]
        24.3         21105829         768   27481.5     1631   160062  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total     Operations  Average   Minimum   Maximum               Operation            
     ----------  ----------  --------  --------  --------  ---------------------------------
     393216.000         192  2048.000  2048.000  2048.000  [CUDA Unified Memory memcpy HtoD]
     131072.000         768   170.667     4.000  1020.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum        Name     
     -------  ---------------  ---------  ----------  -------  ---------  --------------
        79.8       1274388745         69  18469402.1     2354  100130974  poll          
         9.9        158140955        681    232218.7     1039   22553208  ioctl         
         7.8        125032904         57   2193559.7    13781   20810303  sem_timedwait 
         1.6         25390711         94    270113.9     1488    8895883  mmap          
         0.8         12179618          2   6089809.0    41011   12138607  sem_wait      
         0.1          1668057         82     20342.2     4819      40370  open64        
         0.0           216883          3     72294.3    70550      74923  fgets         
         0.0           199047          5     39809.4    32239      53739  pthread_create
         0.0           124619         25      4984.8     1740      19616  fopen         
         0.0           112647         12      9387.3     4251      13515  write         
         0.0            50935         11      4630.5     1828      13674  munmap        
         0.0            28969          5      5793.8     3550       9457  open          
         0.0            28270         18      1570.6     1104       4745  fclose        
         0.0            23903         17      1406.1     1117       3800  fcntl         
         0.0            21582         13      1660.2     1072       2530  read          
         0.0            21571          6      3595.2     1104       9659  fgetc         
         0.0            13691          1     13691.0    13691      13691  pipe2         
         0.0            10171          2      5085.5     4670       5501  socket        
         0.0             9623          3      3207.7     2083       4027  fread         
         0.0             5704          1      5704.0     5704       5704  connect       
         0.0             2150          1      2150.0     2150       2150  bind          
         0.0             1550          1      1550.0     1550       1550  listen        
    
    Report file moved to "/dli/task/vector-add-prefetch-report.qdrep"
    Report file moved to "/dli/task/vector-add-prefetch-report.sqlite"
    


Open the report in Nsight Systems, leaving the previous report open for comparison.

- How does the execution time compare to that of the `addVectorsInto` kernel prior to adding asynchronous prefetching?
- Locate `cudaMemPrefetchAsync` in the *CUDA API* section of the timeline.
- How have the memory transfers changed?


### Exercise: Profile Refactor with Launch Init in Kernel

In the previous iteration of the vector addition application, the vector data is being initialized on the CPU, and therefore needs to be migrated to the GPU before the `addVectorsInto` kernel can operate on it.

The next iteration of the application, [01-init-kernel-solution.cu](../edit/02-init-kernel/solutions/01-init-kernel-solution.cu), the application has been refactored to initialize the data in parallel on the GPU.

Since the initialization now takes place on the GPU, prefetching has been done prior to initialization, rather than prior to the vector addition work. Review the source code to identify where these changes have been made.

After reviewing the changes, compile and run the refactored application using the code execution cell directly below. You should see its success message printed.


```python
!nvcc -o init-kernel 02-init-kernel/solutions/01-init-kernel-solution.cu -run
```

    Success! All values calculated correctly.


Now create a report file for this version of the application:


```python
!nsys profile --stats=true -o init-kernel-report ./init-kernel
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Success! All values calculated correctly.
    Processing events...
    Saving temporary "/tmp/nsys-report-5f32-ace4-b6e6-efee.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-5f32-ace4-b6e6-efee.qdrep"
    Exporting 1859 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-5f32-ace4-b6e6-efee.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum           Name         
     -------  ---------------  ---------  ----------  -------  ---------  ---------------------
        91.5        252124091          3  84041363.7    22287  252048304  cudaMallocManaged    
         6.4         17516020          3   5838673.3   829781   15599715  cudaFree             
         1.3          3558920          1   3558920.0  3558920    3558920  cudaDeviceSynchronize
         0.8          2160427          3    720142.3   689193     759977  cudaMemPrefetchAsync 
         0.0            52789          4     13197.3     5045      35027  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances   Average   Minimum  Maximum                     Name                    
     -------  ---------------  ---------  ---------  -------  -------  -------------------------------------------
        52.3          1867305          3   622435.0   617145   627960  initWith(float, float*, int)               
        47.7          1704011          1  1704011.0  1704011  1704011  addVectorsInto(float*, float*, float*, int)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
     -------  ---------------  ----------  -------  -------  -------  ---------------------------------
       100.0         21213610         768  27621.9     1630   160062  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total     Operations  Average  Minimum  Maximum               Operation            
     ----------  ----------  -------  -------  --------  ---------------------------------
     131072.000         768  170.667    4.000  1020.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum        Name     
     -------  ---------------  ---------  ----------  -------  ---------  --------------
        74.2        572547650         32  17892114.1    24139  100131583  poll          
        14.1        108530709        679    159839.0     1024   18504371  ioctl         
         8.8         68114580         27   2522762.2    21853   20563888  sem_timedwait 
         2.6         20412497         94    217154.2     1420   15532158  mmap          
         0.2          1543347         82     18821.3     5164      34251  open64        
         0.0           215139          3     71713.0    69654      75122  fgets         
         0.0           159600          4     39900.0    34073      45112  pthread_create
         0.0           132350         25      5294.0     1558      25807  fopen         
         0.0            82525         11      7502.3     4169      13477  write         
         0.0            45493         11      4135.7     1508       8834  munmap        
         0.0            33023          5      6604.6     4063       9975  open          
         0.0            28753         18      1597.4     1081       5567  fclose        
         0.0            28142         13      2164.8     1317       4375  read          
         0.0            23888          6      3981.3     1045      10151  fgetc         
         0.0            21889          4      5472.3     2005      12049  fread         
         0.0            14733          8      1841.6     1063       6533  fcntl         
         0.0            13630          2      6815.0     6462       7168  socket        
         0.0             8312          1      8312.0     8312       8312  pipe2         
         0.0             7776          1      7776.0     7776       7776  connect       
         0.0             3028          1      3028.0     3028       3028  bind          
         0.0             1572          1      1572.0     1572       1572  listen        
    
    Report file moved to "/dli/task/init-kernel-report.qdrep"
    Report file moved to "/dli/task/init-kernel-report.sqlite"
    


Open the new report file in Nsight Systems and do the following:

- Compare the application and `addVectorsInto` run times to the previous version of the application, how did they change?
- Look at the *Kernels* section of the timeline. Which of the two kernels (`addVectorsInto` and the initialization kernel) is taking up the majority of the time on the GPU?
- Which of the following does your application contain?
  - Data Migration (HtoD)
  - Data Migration (DtoH)

### Exercise: Profile Refactor with Asynchronous Prefetch Back to the Host

Currently, the vector addition application verifies the work of the vector addition kernel on the host. The next refactor of the application, [01-prefetch-check-solution.cu](../edit/04-prefetch-check/solutions/01-prefetch-check-solution.cu), asynchronously prefetches the data back to the host for verification.

After reviewing the changes, compile and run the refactored application using the code execution cell directly below. You should see its success message printed.


```python
!nvcc -o prefetch-to-host 04-prefetch-check/solutions/01-prefetch-check-solution.cu -run
```

    Success! All values calculated correctly.


Now create a report file for this version of the application:


```python
!nsys profile --stats=true -o prefetch-to-host-report ./prefetch-to-host
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Success! All values calculated correctly.
    Processing events...
    Saving temporary "/tmp/nsys-report-ccc2-31d8-5adf-38a3.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-ccc2-31d8-5adf-38a3.qdrep"
    Exporting 1158 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-ccc2-31d8-5adf-38a3.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum           Name         
     -------  ---------------  ---------  ----------  -------  ---------  ---------------------
        81.2        252404214          3  84134738.0    18510  252336626  cudaMallocManaged    
        14.1         43878374          4  10969593.5   663019   41769636  cudaMemPrefetchAsync 
         3.4         10660288          3   3553429.3   825844    8735064  cudaFree             
         1.2          3720615          1   3720615.0  3720615    3720615  cudaDeviceSynchronize
         0.0            47051          4     11762.8     4583      30360  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances   Average   Minimum  Maximum                     Name                    
     -------  ---------------  ---------  ---------  -------  -------  -------------------------------------------
        52.3          1868937          3   622979.0   619928   624664  initWith(float, float*, int)               
        47.7          1705355          1  1705355.0  1705355  1705355  addVectorsInto(float*, float*, float*, int)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average   Minimum  Maximum              Operation            
     -------  ---------------  ----------  --------  -------  -------  ---------------------------------
       100.0         20436902          64  319326.6   319068   320444  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total     Operations  Average   Minimum   Maximum               Operation            
     ----------  ----------  --------  --------  --------  ---------------------------------
     131072.000          64  2048.000  2048.000  2048.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum        Name     
     -------  ---------------  ---------  ----------  -------  ---------  --------------
        66.1        431214396         27  15970903.6     4223  100128871  poll          
        23.0        149919620        684    219180.7     1008   41692932  ioctl         
         8.5         55186369         24   2299432.0    21688   20879828  sem_timedwait 
         2.1         13671847         94    145445.2     1398    8672313  mmap          
         0.2          1574155         82     19197.0     6394      35079  open64        
         0.0           218311          3     72770.3    69399      76987  fgets         
         0.0           184190          4     46047.5    36837      54460  pthread_create
         0.0           152442         25      6097.7     1623      25245  fopen         
         0.0            96930         11      8811.8     4477      15038  write         
         0.0            50963         12      4246.9     1589       7939  munmap        
         0.0            46298          5      9259.6     4515      16069  open          
         0.0            31548         18      1752.7     1149       5936  fclose        
         0.0            30225         13      2325.0     1444       3626  read          
         0.0            25703          6      4283.8     1085      12284  fgetc         
         0.0            19723          2      9861.5     8124      11599  socket        
         0.0            18406         12      1533.8     1022       5967  fcntl         
         0.0            14393          4      3598.3     2264       5370  fread         
         0.0            10377          1     10377.0    10377      10377  pipe2         
         0.0             9986          1      9986.0     9986       9986  connect       
         0.0             3235          1      3235.0     3235       3235  bind          
         0.0             1879          1      1879.0     1879       1879  listen        
    
    Report file moved to "/dli/task/prefetch-to-host-report.qdrep"
    Report file moved to "/dli/task/prefetch-to-host-report.sqlite"
    


Open this report file in Nsight Systems, and do the following:

- Use the *Unified Memory* section of the timeline to compare and contrast the *Data Migration (DtoH)* events before and after adding prefetching back to the CPU.

---
## Concurrent CUDA Streams

You are now going to learn about a new concept, **CUDA Streams**. After an introduction to them, you will return to using Nsight Systems to better evaluate their impact on your application's performance.

The following slides present upcoming material visually, at a high level. Click through the slides before moving on to more detailed coverage of their topics in following sections.


```python
%%HTML

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/embedded/task3/NVVP-Streams-1.pptx" width="800px" height="500px" frameborder="0"></iframe></div>
```



<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/embedded/task3/NVVP-Streams-1.pptx" width="800px" height="500px" frameborder="0"></iframe></div>



In CUDA programming, a **stream** is a series of commands that execute in order. In CUDA applications, kernel execution, as well as some memory transfers, occur within CUDA streams. Up until this point in time, you have not been interacting explicitly with CUDA streams, but in fact, your CUDA code has been executing its kernels inside of a stream called *the default stream*.

CUDA programmers can create and utilize non-default CUDA streams in addition to the default stream, and in doing so, perform multiple operations, such as executing multiple kernels, concurrently, in different streams. Using multiple streams can add an additional layer of parallelization to your accelerated applications, and offers many more opportunities for application optimization.

### Rules Governing the Behavior of CUDA Streams

There are a few rules, concerning the behavior of CUDA streams, that should be learned in order to utilize them effectively:

- Operations within a given stream occur in order.
- Operations in different non-default streams are not guaranteed to operate in any specific order relative to each other.
- The default stream is blocking and will both wait for all other streams to complete before running, and, will block other streams from running until it completes.

### Creating, Utilizing, and Destroying Non-Default CUDA Streams

The following code snippet demonstrates how to create, utilize, and destroy a non-default CUDA stream. You will note, that to launch a CUDA kernel in a non-default CUDA stream, the stream must be passed as the optional 4th argument of the execution configuration. Up until now you have only utilized the first 2 arguments of the execution configuration:

```cpp
cudaStream_t stream;       // CUDA streams are of type `cudaStream_t`.
cudaStreamCreate(&stream); // Note that a pointer must be passed to `cudaCreateStream`.

someKernel<<<number_of_blocks, threads_per_block, 0, stream>>>(); // `stream` is passed as 4th EC argument.

cudaStreamDestroy(stream); // Note that a value, not a pointer, is passed to `cudaDestroyStream`.
```

Outside the scope of this lab, but worth mentioning, is the optional 3rd argument of the execution configuration. This argument allows programmers to supply the number of bytes in **shared memory** (an advanced topic that will not be covered presently) to be dynamically allocated per block for this kernel launch. The default number of bytes allocated to shared memory per block is `0`, and for the remainder of the lab, you will be passing `0` as this value, in order to expose the 4th argument, which is of immediate interest:

### Exercise: Predict Default Stream Behavior

The [01-print-numbers](../edit/05-stream-intro/01-print-numbers.cu) application has a very simple `printNumber` kernel which accepts an integer and prints it. The kernel is only being executed with a single thread inside a single block. However, it is being executed 5 times, using a for-loop, and passing each launch the number of the for-loop's iteration.

Compile and run [01-print-numbers](../edit/05-stream-intro/01-print-numbers.cu) using the code execution block below. You should see the numbers `0` through `4` printed.


```python
!nvcc -o print-numbers 05-stream-intro/01-print-numbers.cu -run
```

    0
    1
    2
    3
    4


Knowing that by default kernels are executed in the default stream, would you expect that the 5 launches of the `print-numbers` program executed serially, or in parallel? You should be able to mention two features of the default stream to support your answer. Create a report file in the cell below and open it in Nsight Systems to confirm your answer.


```python
!nsys profile --stats=true -o print-numbers-report ./print-numbers
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    0
    1
    2
    3
    4
    Processing events...
    Saving temporary "/tmp/nsys-report-e76f-0d9b-67ab-9e93.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-e76f-0d9b-67ab-9e93.qdrep"
    Exporting 1028 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-e76f-0d9b-67ab-9e93.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum           Name         
     -------  ---------------  ---------  ----------  -------  ---------  ---------------------
        99.9        231590227          5  46318045.4     3972  231570322  cudaLaunchKernel     
         0.1           274340          1    274340.0   274340     274340  cudaDeviceSynchronize
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances  Average  Minimum  Maximum        Name      
     -------  ---------------  ---------  -------  -------  -------  ----------------
       100.0           274364          5  54872.8    53087    61695  printNumber(int)
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum        Name     
     -------  ---------------  ---------  ----------  -------  ---------  --------------
        67.2        230846288         14  16489020.6    24046  100129940  poll          
        30.9        106382468        668    159255.2     1101   18435339  ioctl         
         0.9          3145982         87     36160.7     1479     984888  mmap          
         0.6          1893699         82     23093.9     7394      43232  open64        
         0.2           631128         11     57375.3    14440     384134  sem_timedwait 
         0.1           217219          3     72406.3    69462      75986  fgets         
         0.0           170027          4     42506.8    34711      50220  pthread_create
         0.0           165936         25      6637.4     1585      26231  fopen         
         0.0           103666         12      8638.8     4567      15562  write         
         0.0            36806          5      7361.2     4820      10487  open          
         0.0            31476          7      4496.6     1399       9667  munmap        
         0.0            30858         18      1714.3     1153       5721  fclose        
         0.0            24766          6      4127.7     1070      11018  fgetc         
         0.0            24370         13      1874.6     1013       2926  read          
         0.0            19169          2      9584.5     7529      11640  socket        
         0.0            17964          2      8982.0     5776      12188  fread         
         0.0            13661          9      1517.9     1003       4603  fcntl         
         0.0             8429          1      8429.0     8429       8429  pipe2         
         0.0             7592          1      7592.0     7592       7592  connect       
         0.0             2728          1      2728.0     2728       2728  bind          
         0.0             1577          1      1577.0     1577       1577  listen        
    
    Report file moved to "/dli/task/print-numbers-report.qdrep"
    Report file moved to "/dli/task/print-numbers-report.sqlite"
    


### Exercise: Implement Concurrent CUDA Streams

Both because all 5 kernel launches occurred in the same stream, you should not be surprised to have seen that the 5 kernels executed serially. Additionally you could make the case that because the default stream is blocking, each launch of the kernel would wait to complete before the next launch, and this is also true.

Refactor [01-print-numbers](../edit/05-stream-intro/01-print-numbers.cu) so that each kernel launch occurs in its own non-default stream. Be sure to destroy the streams you create after they are no longer needed. Compile and run the refactored code with the code execution cell directly below. You should still see the numbers `0` through `4` printed, though not necessarily in ascending order. Refer to [the solution](../edit/05-stream-intro/solutions/01-print-numbers-solution.cu) if you get stuck.


```python
!nvcc -o print-numbers-in-streams 05-stream-intro/01-print-numbers.cu -run
```

    0
    1
    2
    3
    4


Now that you are using 5 different non-default streams for each of the 5 kernel launches, do you expect that they will run serially or in parallel? In addition to what you now know about streams, take into account how trivial the `printNumber` kernel is, meaning, even if you predict parallel runs, will the speed at which one kernel will complete allow for complete overlap?

After hypothesizing, open a new report file in Nsight Systems to view its actual behavior. You should notice that now, there are additional rows in the _CUDA_ section for each of the non-default streams you created:


```python
!nsys profile --stats=true -o print-numbers-in-streams-report print-numbers-in-streams
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    0
    1
    2
    3
    4
    Processing events...
    Saving temporary "/tmp/nsys-report-80b0-dda7-f5cd-d58e.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-80b0-dda7-f5cd-d58e.qdrep"
    Exporting 1031 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-80b0-dda7-f5cd-d58e.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum           Name         
     -------  ---------------  ---------  ----------  -------  ---------  ---------------------
        99.9        232188000          5  46437600.0     3983  232166656  cudaLaunchKernel     
         0.1           274624          1    274624.0   274624     274624  cudaDeviceSynchronize
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances  Average  Minimum  Maximum        Name      
     -------  ---------------  ---------  -------  -------  -------  ----------------
       100.0           275516          5  55103.2    53183    62335  printNumber(int)
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum        Name     
     -------  ---------------  ---------  ----------  -------  ---------  --------------
        67.1        231071908         14  16505136.3    24669  100131150  poll          
        31.0        106625128        669    159379.9     1088   18340350  ioctl         
         0.9          3074756         87     35342.0     1324     942128  mmap          
         0.4          1499785         82     18290.1     5654      30185  open64        
         0.3           982950         11     89359.1    22281     659979  sem_timedwait 
         0.1           216922          3     72307.3    69507      75987  fgets         
         0.1           177391          4     44347.8    35740      52742  pthread_create
         0.0           138453         25      5538.1     1542      25362  fopen         
         0.0            93945         12      7828.8     4387      12563  write         
         0.0            65779          9      7308.8     1866      33186  munmap        
         0.0            43884          7      6269.1     1070      16570  fgetc         
         0.0            34449          5      6889.8     4479       9616  open          
         0.0            31744         14      2267.4     1342       3974  read          
         0.0            28994         18      1610.8     1144       5509  fclose        
         0.0            15567          2      7783.5     6599       8968  socket        
         0.0            13499          7      1928.4     1027       6238  fcntl         
         0.0             9418          2      4709.0     3816       5602  fread         
         0.0             8740          1      8740.0     8740       8740  connect       
         0.0             7586          1      7586.0     7586       7586  pipe2         
         0.0             2759          1      2759.0     2759       2759  bind          
         0.0             1730          1      1730.0     1730       1730  listen        
    
    Report file moved to "/dli/task/print-numbers-in-streams-report.qdrep"
    Report file moved to "/dli/task/print-numbers-in-streams-report.sqlite"
    


![streams print](images/streams-print.png)

### Exercise: Use Streams for Concurrent Data Initialization Kernels

The vector addition application you have been working with, [01-prefetch-check-solution.cu](../edit/04-prefetch-check/solutions/01-prefetch-check-solution.cu), currently launches an initialization kernel 3 times - once each for each of the 3 vectors needing initialization for the `vectorAdd` kernel. Refactor it to launch each of the 3 initialization kernel launches in their own non-default stream. You should still see the success message print when compiling and running with the code execution cell below. Refer to [the solution](../edit/06-stream-init/solutions/01-stream-init-solution.cu) if you get stuck.


```python
!nvcc -o init-in-streams 04-prefetch-check/solutions/01-prefetch-check-solution.cu -run
```

    Success! All values calculated correctly.


Open a report in Nsight Systems to confirm that your 3 initialization kernel launches are running in their own non-default streams, with some degree of concurrent overlap.


```python
!nsys profile --stats=true -o init-in-streams-report ./init-in-streams
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Success! All values calculated correctly.
    Processing events...
    Saving temporary "/tmp/nsys-report-8dc5-ecf0-b876-18b7.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-8dc5-ecf0-b876-18b7.qdrep"
    Exporting 1159 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-8dc5-ecf0-b876-18b7.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum           Name         
     -------  ---------------  ---------  ----------  -------  ---------  ---------------------
        80.9        244573306          3  81524435.3    19891  244517730  cudaMallocManaged    
        14.4         43671906          4  10917976.5   668161   41576927  cudaMemPrefetchAsync 
         3.5         10527865          3   3509288.3   842234    8773636  cudaFree             
         1.2          3561958          1   3561958.0  3561958    3561958  cudaDeviceSynchronize
         0.0            41950          4     10487.5     4543      26473  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances   Average   Minimum  Maximum                     Name                    
     -------  ---------------  ---------  ---------  -------  -------  -------------------------------------------
        52.3          1869641          3   623213.7   615769   628600  initWith(float, float*, int)               
        47.7          1702219          1  1702219.0  1702219  1702219  addVectorsInto(float*, float*, float*, int)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average   Minimum  Maximum              Operation            
     -------  ---------------  ----------  --------  -------  -------  ---------------------------------
       100.0         20436450          64  319319.5   319036   320412  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total     Operations  Average   Minimum   Maximum               Operation            
     ----------  ----------  --------  --------  --------  ---------------------------------
     131072.000          64  2048.000  2048.000  2048.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum        Name     
     -------  ---------------  ---------  ----------  -------  ---------  --------------
        67.3        440777481         28  15742052.9    23517  100130959  poll          
        21.9        143514488        682    210431.8     1015   41519041  ioctl         
         8.4         54711008         23   2378739.5    20803   20541787  sem_timedwait 
         2.0         13355760         94    142082.6     1431    8719833  mmap          
         0.3          1674280         82     20418.0     4831      37871  open64        
         0.0           213829          3     71276.3    69211      74445  fgets         
         0.0           148427          4     37106.8    32971      41242  pthread_create
         0.0           116171         25      4646.8     1548      20856  fopen         
         0.0            90438         11      8221.6     4346      13884  write         
         0.0            40755         11      3705.0     1866       5626  munmap        
         0.0            27540         18      1530.0     1109       4290  fclose        
         0.0            27466          5      5493.2     3306       8088  open          
         0.0            23602         16      1475.1     1024       4901  fcntl         
         0.0            22748         13      1749.8     1069       2634  read          
         0.0            21690          6      3615.0     1122       9576  fgetc         
         0.0            12134          4      3033.5     1844       4366  fread         
         0.0            10615          2      5307.5     4884       5731  socket        
         0.0             7620          1      7620.0     7620       7620  pipe2         
         0.0             6242          1      6242.0     6242       6242  connect       
         0.0             2160          1      2160.0     2160       2160  bind          
         0.0             1494          1      1494.0     1494       1494  listen        
    
    Report file moved to "/dli/task/init-in-streams-report.qdrep"
    Report file moved to "/dli/task/init-in-streams-report.sqlite"
    


---
## Summary

At this point in the lab you are able to:

- Use the **Nsight Systems** to visually profile the timeline of GPU-accelerated CUDA applications.
- Use Nsight Systems to identify, and exploit, optimization opportunities in GPU-accelerated CUDA applications.
- Utilize CUDA streams for concurrent kernel execution in accelerated applications.

At this point in time you have a wealth of fundamental tools and techniques for accelerating CPU-only applications, and for then optimizing those accelerated applications. In the final exercise, you will have a chance to apply everything that you've learned to accelerate an [n-body](https://en.wikipedia.org/wiki/N-body_problem) simulator, which predicts the individual motions of a group of objects interacting with each other gravitationally.

---
## Final Exercise: Accelerate and Optimize an N-Body Simulator

An [n-body](https://en.wikipedia.org/wiki/N-body_problem) simulator predicts the individual motions of a group of objects interacting with each other gravitationally. [01-nbody.cu](../edit/09-nbody/01-nbody.cu) contains a simple, though working, n-body simulator for bodies moving through 3 dimensional space.

In its current CPU-only form, this application takes about 5 seconds to run on 4096 particles, and **20 minutes** to run on 65536 particles. Your task is to GPU accelerate the program, retaining the correctness of the simulation.

### Considerations to Guide Your Work

Here are some things to consider before beginning your work:

- Especially for your first refactors, the logic of the application, the `bodyForce` function in particular, can and should remain largely unchanged: focus on accelerating it as easily as possible.
- The code base contains a for-loop inside `main` for integrating the interbody forces calculated by `bodyForce` into the positions of the bodies in the system. This integration both needs to occur after `bodyForce` runs, and, needs to complete before the next call to `bodyForce`. Keep this in mind when choosing how and where to parallelize.
- Use a **profile driven** and iterative approach.
- You are not required to add error handling to your code, but you might find it helpful, as you are responsible for your code working correctly.

**Have Fun!**

Use this cell to compile the nbody simulator. Although it is initially a CPU-only application, is does accurately simulate the positions of the particles.


```python
!nvcc -std=c++11 -o nbody 09-nbody/01-nbody.cu
```

It is highly recommended you use the profiler to assist your work. Execute the following cell to generate a report file:


```python
!nsys profile --stats=true --force-overwrite=true -o nbody-report ./nbody
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    0.041 Billion Interactions / second
    Processing events...
    Saving temporary "/tmp/nsys-report-ec9f-8f8a-c6f2-19fc.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-ec9f-8f8a-c6f2-19fc.qdrep"
    Exporting 39 events: [====================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-ec9f-8f8a-c6f2-19fc.sqlite
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls  Average  Minimum  Maximum   Name  
     -------  ---------------  ---------  -------  -------  -------  -------
        32.5            80266          1  80266.0    80266    80266  writev 
        30.3            74685          2  37342.5     7774    66911  fopen64
        21.7            53575          1  53575.0    53575    53575  read   
        15.5            38284          2  19142.0     2357    35927  fclose 
    
    Report file moved to "/dli/task/nbody-report.qdrep"
    Report file moved to "/dli/task/nbody-report.sqlite"
    


Here we import a function that will run your `nbody` simulator against a various number of particles, checking for performance and accuracy.


```python
from assessment import run_assessment
```

Execute the following cell to run and assess `nbody`:


```python
run_assessment()
```

    Running nbody simulator with 4096 bodies
    ----------------------------------------
    
    Application should run faster than 0.9s
    Your application ran in: 4.1226s
    Your application is not yet fast enough


## Generate a Certificate

If you passed the assessment, please return to the course page (shown below) and click the "ASSESS TASK" button, which will generate your certificate for the course.

![run_assessment](./images/run_assessment.png)

## Advanced Content

The following sections, for those of you with time and interest, introduce more intermediate techniques involving some manual device memory management, and using non-default streams to overlap kernel execution and memory copies.

After learning about each of the techniques below, try to further optimize your nbody simulation using these techniques.

---
## Manual Device Memory Allocation and Copying

While `cudaMallocManaged` and `cudaMemPrefetchAsync` are performant, and greatly simplify memory migration, sometimes it can be worth it to use more manual methods for memory allocation. This is particularly true when it is known that data will only be accessed on the device or host, and the cost of migrating data can be reclaimed in exchange for the fact that no automatic on-demand migration is needed.

Additionally, using manual device memory management can allow for the use of non-default streams for overlapping data transfers with computational work. In this section you will learn some basic manual device memory allocation and copy techniques, before extending these techniques to overlap data copies with computational work. 

Here are some CUDA commands for manual device memory management:

- `cudaMalloc` will allocate memory directly to the active GPU. This prevents all GPU page faults. In exchange, the pointer it returns is not available for access by host code.
- `cudaMallocHost` will allocate memory directly to the CPU. It also "pins" the memory, or page locks it, which will allow for asynchronous copying of the memory to and from a GPU. Too much pinned memory can interfere with CPU performance, so use it only with intention. Pinned memory should be freed with `cudaFreeHost`.
- `cudaMemcpy` can copy (not transfer) memory, either from host to device or from device to host.

### Manual Device Memory Management Example

Here is a snippet of code that demonstrates the use of the above CUDA API calls.

```cpp
int *host_a, *device_a;        // Define host-specific and device-specific arrays.
cudaMalloc(&device_a, size);   // `device_a` is immediately available on the GPU.
cudaMallocHost(&host_a, size); // `host_a` is immediately available on CPU, and is page-locked, or pinned.

initializeOnHost(host_a, N);   // No CPU page faulting since memory is already allocated on the host.

// `cudaMemcpy` takes the destination, source, size, and a CUDA-provided variable for the direction of the copy.
cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

kernel<<<blocks, threads, 0, someStream>>>(device_a, N);

// `cudaMemcpy` can also copy data from device to host.
cudaMemcpy(host_a, device_a, size, cudaMemcpyDeviceToHost);

verifyOnHost(host_a, N);

cudaFree(device_a);
cudaFreeHost(host_a);          // Free pinned memory like this.
```

### Exercise: Manually Allocate Host and Device Memory

The most recent iteration of the vector addition application, [01-stream-init-solution](../edit/06-stream-init/solutions/01-stream-init-solution.cu), is using `cudaMallocManaged` to allocate managed memory first used on the device by the initialization kernels, then on the device by the vector add kernel, and then by the host, where the memory is automatically transferred, for verification. This is a sensible approach, but it is worth experimenting with some manual device memory allocation and copying to observe its impact on the application's performance.

Refactor the [01-stream-init-solution](../edit/06-stream-init/solutions/01-stream-init-solution.cu) application to **not** use `cudaMallocManaged`. In order to do this you will need to do the following:

- Replace calls to `cudaMallocManaged` with `cudaMalloc`.
- Create an additional vector that will be used for verification on the host. This is required since the memory allocated with `cudaMalloc` is not available to the host. Allocate this host vector with `cudaMallocHost`.
- After the `addVectorsInto` kernel completes, use `cudaMemcpy` to copy the vector with the addition results, into the host vector you created with `cudaMallocHost`.
- Use `cudaFreeHost` to free the memory allocated with `cudaMallocHost`.

Refer to [the solution](../edit/07-manual-malloc/solutions/01-manual-malloc-solution.cu) if you get stuck.


```python
!nvcc -o vector-add-manual-alloc 06-stream-init/solutions/01-stream-init-solution.cu -run
```

After completing the refactor, open a report in Nsight Systems, and use the timeline to do the following:

- Notice that there is no longer a *Unified Memory* section of the timeline.
- Comparing this timeline to that of the previous refactor, compare the run times of `cudaMalloc` in the current application vs. `cudaMallocManaged` in the previous.
- Notice how in the current application, work on the initialization kernels does not start until a later time than it did in the previous iteration. Examination of the timeline will show the difference is the time taken by `cudaMallocHost`. This clearly points out the difference between memory transfers, and memory copies. When copying memory, as you are doing presently, the data will exist in 2 different places in the system. In the current case, the allocation of the 4th host-only vector incurs a small cost in performance, compared to only allocating 3 vectors in the previous iteration.

---
## Using Streams to Overlap Data Transfers and Code Execution

The following slides present upcoming material visually, at a high level. Click through the slides before moving on to more detailed coverage of their topics in following sections.


```python
%%HTML

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/embedded/task3/NVVP-Streams-3.pptx" width="800px" height="500px" frameborder="0"></iframe></div>
```

In addition to `cudaMemcpy` is `cudaMemcpyAsync` which can asynchronously copy memory either from host to device or from device to host as long as the host memory is pinned, which can be done by allocating it with `cudaMallocHost`.

Similar to kernel execution, `cudaMemcpyAsync` is only asynchronous by default with respect to the host. It executes, by default, in the default stream and therefore is a blocking operation with regard to other CUDA operations occurring on the GPU. The `cudaMemcpyAsync` function, however, takes as an optional 5th argument, a non-default stream. By passing it a non-default stream, the memory transfer can be concurrent to other CUDA operations occurring in other non-default streams.

A common and useful pattern is to use a combination of pinned host memory, asynchronous memory copies in non-default streams, and kernel executions in non-default streams, to overlap memory transfers with kernel execution.

In the following example, rather than wait for the entire memory copy to complete before beginning work on the kernel, segments of the required data are copied and worked on, with each copy/work segment running in its own non-default stream. Using this technique, work on parts of the data can begin while memory transfers for later segments occur concurrently. Extra care must be taken when using this technique to calculate segment-specific values for the number of operations, and the offset location inside arrays, as shown here:

```cpp
int N = 2<<24;
int size = N * sizeof(int);

int *host_array;
int *device_array;

cudaMallocHost(&host_array, size);               // Pinned host memory allocation.
cudaMalloc(&device_array, size);                 // Allocation directly on the active GPU device.

initializeData(host_array, N);                   // Assume this application needs to initialize on the host.

const int numberOfSegments = 4;                  // This example demonstrates slicing the work into 4 segments.
int segmentN = N / numberOfSegments;             // A value for a segment's worth of `N` is needed.
size_t segmentSize = size / numberOfSegments;    // A value for a segment's worth of `size` is needed.

// For each of the 4 segments...
for (int i = 0; i < numberOfSegments; ++i)
{
  // Calculate the index where this particular segment should operate within the larger arrays.
  segmentOffset = i * segmentN;

  // Create a stream for this segment's worth of copy and work.
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Asynchronously copy segment's worth of pinned host memory to device over non-default stream.
  cudaMemcpyAsync(&device_array[segmentOffset],  // Take care to access correct location in array.
                  &host_array[segmentOffset],    // Take care to access correct location in array.
                  segmentSize,                   // Only copy a segment's worth of memory.
                  cudaMemcpyHostToDevice,
                  stream);                       // Provide optional argument for non-default stream.
                  
  // Execute segment's worth of work over same non-default stream as memory copy.
  kernel<<<number_of_blocks, threads_per_block, 0, stream>>>(&device_array[segmentOffset], segmentN);
  
  // `cudaStreamDestroy` will return immediately (is non-blocking), but will not actually destroy stream until
  // all stream operations are complete.
  cudaStreamDestroy(stream);
}
```

### Exercise: Overlap Kernel Execution and Memory Copy Back to Host

The most recent iteration of the vector addition application, [01-manual-malloc-solution.cu](../edit/07-manual-malloc/solutions/01-manual-malloc-solution.cu), is currently performing all of its vector addition work on the GPU before copying the memory back to the host for verification.

Refactor [01-manual-malloc-solution.cu](../edit/07-manual-malloc/solutions/01-manual-malloc-solution.cu) to perform the vector addition in 4 segments, in non-default streams, so that asynchronous memory copies can begin before waiting for all vector addition work to complete. Refer to [the solution](../edit/08-overlap-xfer/solutions/01-overlap-xfer-solution.cu) if you get stuck.


```python
!nvcc -o vector-add-manual-alloc 07-manual-malloc/solutions/01-manual-malloc-solution.cu -run
```

After completing the refactor, open a report in Nsight Systems, and use the timeline to do the following:

- Note when the device to host memory transfers begin, is it before or after all kernel work has completed?
- Notice that the 4 memory copy segments themselves do not overlap. Even in separate non-default streams, only one memory transfer in a given direction (DtoH here) at a time can occur simultaneously. The performance gains here are in the ability to start the transfers earlier than otherwise, and it is not hard to imagine in an application where a less trivial amount of work was being done compared to a simple addition operation, that the memory copies would not only start earlier, but also overlap with kernel execution.
