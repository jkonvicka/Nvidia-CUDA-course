<h1><div align="center">Managing Accelerated Application Memory with CUDA C/C++ Unified Memory</div></h1>

![CUDA](./images/CUDA_Logo.jpg)

The [*CUDA Best Practices Guide*](http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations), a highly recommended followup to this and other CUDA fundamentals labs, recommends a design cycle called **APOD**: **A**ssess, **P**arallelize, **O**ptimize, **D**eploy. In short, APOD prescribes an iterative design process, where developers can apply incremental improvements to their accelerated application's performance, and ship their code. As developers become more competent CUDA programmers, more advanced optimization techniques can be applied to their accelerated code bases.

This lab will support such a style of iterative development. You will be using the Nsight Systems command line tool **nsys** to qualitatively measure your application's performance, and to identify opportunities for optimization, after which you will apply incremental improvements before learning new techniques and repeating the cycle. As a point of focus, many of the techniques you will be learning and applying in this lab will deal with the specifics of how CUDA's **Unified Memory** works. Understanding Unified Memory behavior is a fundamental skill for CUDA developers, and serves as a prerequisite to many more advanced memory management techniques.

---
## Prerequisites

To get the most out of this lab you should already be able to:

- Write, compile, and run C/C++ programs that both call CPU functions and launch GPU kernels.
- Control parallel thread hierarchy using execution configuration.
- Refactor serial loops to execute their iterations in parallel on a GPU.
- Allocate and free Unified Memory.

---
## Objectives

By the time you complete this lab, you will be able to:

- Use the Nsight Systems command line tool (**nsys**) to profile accelerated application performance.
- Leverage an understanding of **Streaming Multiprocessors** to optimize execution configurations.
- Understand the behavior of **Unified Memory** with regard to page faulting and data migrations.
- Use **asynchronous memory prefetching** to reduce page faults and data migrations for increased performance.
- Employ an iterative development cycle to rapidly accelerate and deploy applications.

---
## Iterative Optimizations with the NVIDIA Command Line Profiler

The only way to be assured that attempts at optimizing accelerated code bases are actually successful is to profile the application for quantitative information about the application's performance. `nsys` is the Nsight Systems command line tool. It ships with the CUDA toolkit, and is a powerful tool for profiling accelerated applications.

`nsys` is easy to use. Its most basic usage is to simply pass it the path to an executable compiled with `nvcc`. `nsys` will proceed to execute the application, after which it will print a summary output of the application's GPU activities, CUDA API calls, as well as information about **Unified Memory** activity, a topic which will be covered extensively later in this lab.

When accelerating applications, or optimizing already-accelerated applications, take a scientific and iterative approach. Profile your application after making changes, take note, and record the implications of any refactoring on performance. Make these observations early and often: frequently, enough performance boost can be gained with little effort such that you can ship your accelerated application. Additionally, frequent profiling will teach you how specific changes to your CUDA code bases impact its actual performance: knowledge that is hard to acquire when only profiling after many kinds of changes in your code bases.

### Exercise: Profile an Application with nsys

[01-vector-add.cu](../edit/01-vector-add/01-vector-add.cu) (<------ you can click on this and any of the source file links in this lab to open them for editing) is a naively accelerated vector addition program. Use the two code execution cells below (`CTRL` + `ENTER`). The first code execution cell will compile (and run) the vector addition program. The second code execution cell will profile the executable that was just compiled using `nsys profile`.

`nsys profile` will generate a `qdrep` report file which can be used in a variety of manners. We use the `--stats=true` flag here to indicate we would like summary statistics printed. There is quite a lot of information printed:

- Profile configuration details
- Report file(s) generation details
- **CUDA API Statistics**
- **CUDA Kernel Statistics**
- **CUDA Memory Operation Statistics (time and size)**
- OS Runtime API Statistics

In this lab you will primarily be using the 3 sections in **bold** above. In the next lab, you will be using the generated report files to give to the Nsight Systems GUI for visual profiling.

After profiling the application, answer the following questions using information displayed in the `CUDA Kernel Statistics` section of the profiling output:

- What was the name of the only CUDA kernel called in this application?
- How many times did this kernel run?
- How long did it take this kernel to run? Record this time somewhere: you will be optimizing this application and will want to know how much faster you can make it.


```python
!nvcc -o single-thread-vector-add 01-vector-add/01-vector-add.cu -run
```

    Success! All values calculated correctly.



```python
!nsys profile --stats=true ./single-thread-vector-add
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Success! All values calculated correctly.
    Processing events...
    Saving temporary "/tmp/nsys-report-d5db-43ad-daa7-d537.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-d5db-43ad-daa7-d537.qdrep"
    Exporting 4660 events: [==================================================100%]                                               ]
    
    Exported successfully to
    /tmp/nsys-report-d5db-43ad-daa7-d537.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls    Average      Minimum     Maximum            Name         
     -------  ---------------  ---------  ------------  ----------  ----------  ---------------------
        88.5       2298324729          1  2298324729.0  2298324729  2298324729  cudaDeviceSynchronize
        10.5        272427936          3    90809312.0       20844   272333765  cudaMallocManaged    
         1.0         24994527          3     8331509.0     7648836     9156888  cudaFree             
         0.0           781703          1      781703.0      781703      781703  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances    Average      Minimum     Maximum                       Name                    
     -------  ---------------  ---------  ------------  ----------  ----------  -------------------------------------------
       100.0       2299046069          1  2299046069.0  2299046069  2299046069  addVectorsInto(float*, float*, float*, int)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
     -------  ---------------  ----------  -------  -------  -------  ---------------------------------
        76.6         68477200        2304  29721.0     1855   182046  [CUDA Unified Memory memcpy HtoD]
        23.4         20958699         768  27290.0     1150   163870  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total     Operations  Average  Minimum  Maximum               Operation            
     ----------  ----------  -------  -------  --------  ---------------------------------
     393216.000        2304  170.667    4.000  1020.000  [CUDA Unified Memory memcpy HtoD]
     131072.000         768  170.667    4.000  1020.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum              Name           
     -------  ---------------  ---------  ----------  -------  ---------  --------------------------
        89.0       5327246775        272  19585466.1    52651  100135165  poll                      
         8.5        506478442        241   2101570.3    16177   20875768  sem_timedwait             
         2.1        123452185        682    181014.9     1018   32102213  ioctl                     
         0.5         27942239         98    285124.9     1600    9077941  mmap                      
         0.0          2108327         82     25711.3     9027      58798  open64                    
         0.0           455826          3    151942.0   138414     165863  fgets                     
         0.0           323090          4     80772.5    60683      99203  pthread_create            
         0.0           225672         25      9026.9     2887      41426  fopen                     
         0.0           120669         11     10969.9     4703      17490  write                     
         0.0            92427          8     11553.4     1081      16645  pthread_rwlock_timedwrlock
         0.0            65949          7      9421.3     1826      27205  fgetc                     
         0.0            64887          5     12977.4     6371      19252  open                      
         0.0            63172         11      5742.9     2827      12657  munmap                    
         0.0            58996         41      1438.9     1027       4971  fcntl                     
         0.0            47278         18      2626.6     1534       4944  fclose                    
         0.0            37224          2     18612.0    17463      19761  socket                    
         0.0            28844          1     28844.0    28844      28844  sem_wait                  
         0.0            26853         12      2237.8     1016       5371  read                      
         0.0            18178          3      6059.3     1579      11299  fread                     
         0.0            17264          1     17264.0    17264      17264  connect                   
         0.0            16213          4      4053.3     3242       5308  mprotect                  
         0.0             9471          1      9471.0     9471       9471  pipe2                     
         0.0             6270          1      6270.0     6270       6270  bind                      
         0.0             4231          1      4231.0     4231       4231  listen                    
    
    Report file moved to "/dli/task/report1.qdrep"
    Report file moved to "/dli/task/report1.sqlite"
    


Worth mentioning is that by default, `nsys profile` will not overwrite an existing report file. This is done to prevent accidental loss of work when profiling. If for any reason, you would rather overwrite an existing report file, say during rapid iterations, you can provide the `-f` flag to `nsys profile` to allow overwriting an existing report file.

### Exercise: Optimize and Profile

Take a minute or two to make a simple optimization to [01-vector-add.cu](../edit/01-vector-add/01-vector-add.cu) by updating its execution configuration so that it runs on many threads in a single thread block. Recompile and then profile with `nsys profile --stats=true` using the code execution cells below. Use the profiling output to check the runtime of the kernel. What was the speed up from this optimization? Be sure to record your results somewhere.


```python
!nvcc -o multi-thread-vector-add 01-vector-add/01-vector-add.cu -run
```

    Success! All values calculated correctly.



```python
!nsys profile --stats=true ./multi-thread-vector-add
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Success! All values calculated correctly.
    Processing events...
    Saving temporary "/tmp/nsys-report-45ef-43a9-8762-4915.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-45ef-43a9-8762-4915.qdrep"
    Exporting 4261 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-45ef-43a9-8762-4915.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls    Average     Minimum    Maximum           Name         
     -------  ---------------  ---------  -----------  ---------  ---------  ---------------------
        54.3        246619849          3   82206616.3      18446  246537316  cudaMallocManaged    
        40.5        183944456          1  183944456.0  183944456  183944456  cudaDeviceSynchronize
         5.1         23370869          3    7790289.7    7049896    9142018  cudaFree             
         0.0            44111          1      44111.0      44111      44111  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances    Average     Minimum    Maximum                      Name                    
     -------  ---------------  ---------  -----------  ---------  ---------  -------------------------------------------
       100.0        183935222          1  183935222.0  183935222  183935222  addVectorsInto(float*, float*, float*, int)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
     -------  ---------------  ----------  -------  -------  -------  ---------------------------------
        76.8         69343442        2304  30097.0     2175   183198  [CUDA Unified Memory memcpy HtoD]
        23.2         20991098         768  27332.2     1599   164990  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total     Operations  Average  Minimum  Maximum               Operation            
     ----------  ----------  -------  -------  --------  ---------------------------------
     393216.000        2304  170.667    4.000  1020.000  [CUDA Unified Memory memcpy HtoD]
     131072.000         768  170.667    4.000  1020.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum              Name           
     -------  ---------------  ---------  ----------  -------  ---------  --------------------------
        84.3       1474900181         77  19154547.8    31192  100130136  poll                      
         8.5        148027327         67   2209363.1    15614   20863360  sem_timedwait             
         5.6         98761127        677    145880.5     1008   17658506  ioctl                     
         1.5         25902440         98    264310.6     1434    9055027  mmap                      
         0.1          1903352         82     23211.6     4692      42975  open64                    
         0.0           205787          3     68595.7    65710      72646  fgets                     
         0.0           174533          4     43633.3    31997      54002  pthread_create            
         0.0           132056         25      5282.2     1520      24163  fopen                     
         0.0            96551         11      8777.4     4120      13592  write                     
         0.0            51908         11      4718.9     1689       8451  munmap                    
         0.0            34709          3     11569.7     6955      16701  pthread_rwlock_timedwrlock
         0.0            30896         23      1343.3     1000       5069  fcntl                     
         0.0            29084          5      5816.8     3254       8687  open                      
         0.0            27465         18      1525.8     1037       4664  fclose                    
         0.0            21890          6      3648.3     1035       9366  fgetc                     
         0.0            20273         12      1689.4     1011       2864  read                      
         0.0            14510          2      7255.0     5897       8613  socket                    
         0.0            10890          3      3630.0     1572       5161  fread                     
         0.0             9074          1      9074.0     9074       9074  pipe2                     
         0.0             8554          4      2138.5     1757       2651  mprotect                  
         0.0             6928          1      6928.0     6928       6928  connect                   
         0.0             2569          1      2569.0     2569       2569  bind                      
         0.0             1817          1      1817.0     1817       1817  listen                    
    
    Report file moved to "/dli/task/report2.qdrep"
    Report file moved to "/dli/task/report2.sqlite"
    


### Exercise: Optimize Iteratively

In this exercise you will go through several cycles of editing the execution configuration of [01-vector-add.cu](../edit/01-vector-add/01-vector-add.cu), profiling it, and recording the results to see the impact. Use the following guidelines while working:

- Start by listing 3 to 5 different ways you will update the execution configuration, being sure to cover a range of different grid and block size combinations.
- Edit the [01-vector-add.cu](../edit/01-vector-add/01-vector-add.cu) program in one of the ways you listed.
- Compile and profile your updated code with the two code execution cells below.
- Record the runtime of the kernel execution, as given in the profiling output.
- Repeat the edit/profile/record cycle for each possible optimization you listed above

Which of the execution configurations you attempted proved to be the fastest?


```python
!nvcc -o iteratively-optimized-vector-add 01-vector-add/01-vector-add.cu -run
```

    Success! All values calculated correctly.



```python
!nsys profile --stats=true ./iteratively-optimized-vector-add
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Success! All values calculated correctly.
    Processing events...
    Saving temporary "/tmp/nsys-report-6ca1-38a9-6593-a45c.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-6ca1-38a9-6593-a45c.qdrep"
    Exporting 4247 events: [==================================================100%][3%                                                    ]
    
    Exported successfully to
    /tmp/nsys-report-6ca1-38a9-6593-a45c.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls    Average     Minimum    Maximum           Name         
     -------  ---------------  ---------  -----------  ---------  ---------  ---------------------
        53.6        245300981          3   81766993.7      18286  245216109  cudaMallocManaged    
        41.3        189063654          1  189063654.0  189063654  189063654  cudaDeviceSynchronize
         5.1         23281049          3    7760349.7    6900701    9316795  cudaFree             
         0.0            45564          1      45564.0      45564      45564  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances    Average     Minimum    Maximum                      Name                    
     -------  ---------------  ---------  -----------  ---------  ---------  -------------------------------------------
       100.0        189054043          1  189054043.0  189054043  189054043  addVectorsInto(float*, float*, float*, int)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
     -------  ---------------  ----------  -------  -------  -------  ---------------------------------
        76.7         69241553        2304  30052.8     2174   182654  [CUDA Unified Memory memcpy HtoD]
        23.3         21023601         768  27374.5     1566   160031  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total     Operations  Average  Minimum  Maximum               Operation            
     ----------  ----------  -------  -------  --------  ---------------------------------
     393216.000        2304  170.667    4.000  1020.000  [CUDA Unified Memory memcpy HtoD]
     131072.000         768  170.667    4.000  1020.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum              Name           
     -------  ---------------  ---------  ----------  -------  ---------  --------------------------
        84.4       1475303506         77  19159785.8    30829  100134307  poll                      
         8.4        146384038         67   2184836.4    14907   20878709  sem_timedwait             
         5.6         98159861        676    145206.9     1011   17904745  ioctl                     
         1.5         25839380         98    263667.1     1396    9247562  mmap                      
         0.1          1858411         82     22663.5     5998      36413  open64                    
         0.0           220863          3     73621.0    68043      79154  fgets                     
         0.0           173939          4     43484.8    33163      58087  pthread_create            
         0.0           142199         25      5688.0     1525      24600  fopen                     
         0.0            87159         11      7923.5     4605      13339  write                     
         0.0            53104         12      4425.3     1303       8616  munmap                    
         0.0            32924          5      6584.8     3260       9660  open                      
         0.0            28112         18      1561.8     1033       4953  fclose                    
         0.0            23728          3      7909.3     5022      11700  pthread_rwlock_timedwrlock
         0.0            23432          5      4686.4     1118      11426  fgetc                     
         0.0            22649         13      1742.2     1056       2938  read                      
         0.0            16995          2      8497.5     6767      10228  socket                    
         0.0            14084          9      1564.9     1028       4747  fcntl                     
         0.0            11205          3      3735.0     1721       5335  fread                     
         0.0             9285          1      9285.0     9285       9285  pipe2                     
         0.0             8558          4      2139.5     1809       2947  mprotect                  
         0.0             8278          1      8278.0     8278       8278  connect                   
         0.0             2644          1      2644.0     2644       2644  bind                      
         0.0             1639          1      1639.0     1639       1639  listen                    
    
    Report file moved to "/dli/task/report3.qdrep"
    Report file moved to "/dli/task/report3.sqlite"
    


---
## Streaming Multiprocessors and Querying the Device

This section explores how understanding a specific feature of the GPU hardware can promote optimization. After introducing **Streaming Multiprocessors**, you will attempt to further optimize the accelerated vector addition program you have been working on.

The following slides present upcoming material visually, at a high level. Click through the slides before moving on to more detailed coverage of their topics in following sections.


```python
%%HTML

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/embedded/task2/NVPROF_UM_1.pptx" width="800px" height="500px" frameborder="0"></iframe></div>
```



<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/embedded/task2/NVPROF_UM_1.pptx" width="800px" height="500px" frameborder="0"></iframe></div>



### Streaming Multiprocessors and Warps

The GPUs that CUDA applications run on have processing units called **streaming multiprocessors**, or **SMs**. During kernel execution, blocks of threads are given to SMs to execute. In order to support the GPU's ability to perform as many parallel operations as possible, performance gains can often be had by *choosing a grid size that has a number of blocks that is a multiple of the number of SMs on a given GPU.*

Additionally, SMs create, manage, schedule, and execute groupings of 32 threads from within a block called **warps**. A more [in depth coverage of SMs and warps](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation) is beyond the scope of this course, however, it is important to know that performance gains can also be had by *choosing a block size that has a number of threads that is a multiple of 32.*

### Programmatically Querying GPU Device Properties

In order to support portability, since the number of SMs on a GPU can differ depending on the specific GPU being used, the number of SMs should not be hard-coded into a code bases. Rather, this information should be acquired programatically.

The following shows how, in CUDA C/C++, to obtain a C struct which contains many properties about the currently active GPU device, including its number of SMs:

```cpp
int deviceId;
cudaGetDevice(&deviceId);                  // `deviceId` now points to the id of the currently active GPU.

cudaDeviceProp props;
cudaGetDeviceProperties(&props, deviceId); // `props` now has many useful properties about
                                           // the active GPU device.
```

### Exercise: Query the Device

Currently, [`01-get-device-properties.cu`](../edit/04-device-properties/01-get-device-properties.cu) contains many unassigned variables, and will print gibberish information intended to describe details about the currently active GPU.

Build out [`01-get-device-properties.cu`](../edit/04-device-properties/01-get-device-properties.cu) to print the actual values for the desired device properties indicated in the source code. In order to support your work, and as an introduction to them, use the [CUDA Runtime Docs](http://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html) to help identify the relevant properties in the device props struct. Refer to [the solution](../edit/04-device-properties/solutions/01-get-device-properties-solution.cu) if you get stuck.


```python
!nvcc -o get-device-properties 04-device-properties/01-get-device-properties.cu -run
```

    Device ID: 0
    Number of SMs: 40
    Compute Capability Major: 7
    Compute Capability Minor: 5
    Warp Size: 32


### Exercise: Optimize Vector Add with Grids Sized to Number of SMs

Utilize your ability to query the device for its number of SMs to refactor the `addVectorsInto` kernel you have been working on inside [01-vector-add.cu](../edit/01-vector-add/01-vector-add.cu) so that it launches with a grid containing a number of blocks that is a multiple of the number of SMs on the device.

Depending on other specific details in the code you have written, this refactor may or may not improve, or significantly change, the performance of your kernel. Therefore, as always, be sure to use `nsys profile` so that you can quantitatively evaluate performance changes. Record the results with the rest of your findings thus far, based on the profiling output.


```python
!nvcc -o sm-optimized-vector-add 01-vector-add/01-vector-add.cu -run
```


```python
!nsys profile --stats=true ./sm-optimized-vector-add
```

---
## Unified Memory Details

You have been allocating memory intended for use either by host or device code with `cudaMallocManaged` and up until now have enjoyed the benefits of this method - automatic memory migration, ease of programming - without diving into the details of how the **Unified Memory** (**UM**) allocated by `cudaMallocManaged` actual works.

`nsys profile` provides details about UM management in accelerated applications, and using this information, in conjunction with a more-detailed understanding of how UM works, provides additional opportunities to optimize accelerated applications.

The following slides present upcoming material visually, at a high level. Click through the slides before moving on to more detailed coverage of their topics in following sections.


```python
%%HTML

<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/embedded/task2/NVPROF_UM_2.pptx" width="800px" height="500px" frameborder="0"></iframe></div>
```



<div align="center"><iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://developer.download.nvidia.com/training/courses/C-AC-01-V1/embedded/task2/NVPROF_UM_2.pptx" width="800px" height="500px" frameborder="0"></iframe></div>



### Unified Memory Migration

When UM is allocated, the memory is not resident yet on either the host or the device. When either the host or device attempts to access the memory, a [page fault](https://en.wikipedia.org/wiki/Page_fault) will occur, at which point the host or device will migrate the needed data in batches. Similarly, at any point when the CPU, or any GPU in the accelerated system, attempts to access memory not yet resident on it, page faults will occur and trigger its migration.

The ability to page fault and migrate memory on demand is tremendously helpful for ease of development in your accelerated applications. Additionally, when working with data that exhibits sparse access patterns, for example when it is impossible to know which data will be required to be worked on until the application actually runs, and for scenarios when data might be accessed by multiple GPU devices in an accelerated system with multiple GPUs, on-demand memory migration is remarkably beneficial.

There are times - for example when data needs are known prior to runtime, and large contiguous blocks of memory are required - when the overhead of page faulting and migrating data on demand incurs an overhead cost that would be better avoided.

Much of the remainder of this lab will be dedicated to understanding on-demand migration, and how to identify it in the profiler's output. With this knowledge you will be able to reduce the overhead of it in scenarios when it would be beneficial.

### Exercise: Explore UM Migration and Page Faulting

`nsys profile` provides output describing UM behavior for the profiled application. In this exercise, you will make several modifications to a simple application, and make use of `nsys profile` after each change, to explore how UM data migration behaves.

[`01-page-faults.cu`](../edit/06-unified-memory-page-faults/01-page-faults.cu) contains a `hostFunction` and a `gpuKernel`, both which could be used to initialize the elements of a `2<<24` element vector with the number `1`. Currently neither the host function nor GPU kernel are being used.

For each of the 4 questions below, given what you have just learned about UM behavior, first hypothesize about what kind of page faulting should happen, then, edit [`01-page-faults.cu`](../edit/06-unified-memory-page-faults/01-page-faults.cu) to create a scenario, by using one or both of the 2 provided functions in the code bases, that will allow you to test your hypothesis.

In order to test your hypotheses, compile and profile your code using the code execution cells below. Be sure to record your hypotheses, as well as the results, obtained from `nsys profile --stats=true` output. In the output of `nsys profile --stats=true` you should be looking for the following:

- Is there a _CUDA Memory Operation Statistics_ section in the output?
- If so, does it indicate host to device (HtoD) or device to host (DtoH) migrations?
- When there are migrations, what does the output say about how many _Operations_ there were? If you see many small memory migration operations, this is a sign that on-demand page faulting is occurring, with small memory migrations occurring each time there is a page fault in the requested location.

Here are the scenarios for you to explore, along with solutions for them if you get stuck:

- Is there evidence of memory migration and/or page faulting when unified memory is accessed only by the CPU? ([solution](../edit/06-unified-memory-page-faults/solutions/01-page-faults-solution-cpu-only.cu))
- Is there evidence of memory migration and/or page faulting when unified memory is accessed only by the GPU? ([solution](../edit/06-unified-memory-page-faults/solutions/02-page-faults-solution-gpu-only.cu))
- Is there evidence of memory migration and/or page faulting when unified memory is accessed first by the CPU then the GPU? ([solution](../edit/06-unified-memory-page-faults/solutions/03-page-faults-solution-cpu-then-gpu.cu))
- Is there evidence of memory migration and/or page faulting when unified memory is accessed first by the GPU then the CPU? ([solution](../edit/06-unified-memory-page-faults/solutions/04-page-faults-solution-gpu-then-cpu.cu))


```python
!nvcc -o page-faults 06-unified-memory-page-faults/01-page-faults.cu -run
```


```python
!nsys profile --stats=true ./page-faults
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Processing events...
    Saving temporary "/tmp/nsys-report-7040-6ab7-c251-5a01.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-7040-6ab7-c251-5a01.qdrep"
    Exporting 1035 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-7040-6ab7-c251-5a01.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls    Average     Minimum    Maximum         Name       
     -------  ---------------  ---------  -----------  ---------  ---------  -----------------
       100.0        241724032          1  241724032.0  241724032  241724032  cudaMallocManaged
         0.0            42031          1      42031.0      42031      42031  cudaFree         
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum              Name           
     -------  ---------------  ---------  ----------  -------  ---------  --------------------------
        64.0        239719454         17  14101144.4    38476  100123124  poll                      
        25.5         95704846        666    143701.0     1002   17786045  ioctl                     
         9.2         34450596         14   2460756.9    16235   20785801  sem_timedwait             
         0.7          2720156         92     29566.9     1472     786899  mmap                      
         0.4          1432313         82     17467.2     4643      35264  open64                    
         0.1           212803          3     70934.3    69668      72868  fgets                     
         0.0           141854          4     35463.5    30294      44902  pthread_create            
         0.0           113400         25      4536.0     1621      20759  fopen                     
         0.0            77211         11      7019.2     4453      10987  write                     
         0.0            66775          5     13355.0     9393      16773  pthread_rwlock_timedwrlock
         0.0            40871          8      5108.9     1056      11286  fgetc                     
         0.0            27540         18      1530.0     1079       3565  fclose                    
         0.0            26804          7      3829.1     2166       5782  munmap                    
         0.0            25951          5      5190.2     3282       7314  open                      
         0.0            23868         13      1836.0     1112       3452  read                      
         0.0            13236          2      6618.0     4817       8419  fread                     
         0.0            11449          7      1635.6     1055       4552  fcntl                     
         0.0             9276          2      4638.0     4377       4899  socket                    
         0.0             7837          1      7837.0     7837       7837  pipe2                     
         0.0             7329          1      7329.0     7329       7329  connect                   
         0.0             7049          4      1762.3     1712       1809  mprotect                  
         0.0             2066          1      2066.0     2066       2066  bind                      
         0.0             1503          1      1503.0     1503       1503  listen                    
    
    Report file moved to "/dli/task/report4.qdrep"
    Report file moved to "/dli/task/report4.sqlite"
    


### Exercise: Revisit UM Behavior for Vector Add Program

Returning to the [01-vector-add.cu](../edit/01-vector-add/01-vector-add.cu) program you have been working on throughout this lab, review the code bases in its current state, and hypothesize about what kinds of memory migrations and/or page faults you expect to occur. Look at the profiling output for your last refactor (either by scrolling up to find the output or by executing the code execution cell just below), observing the _CUDA Memory Operation Statistics_ section of the profiler output. Can you explain the kinds of migrations and the number of their operations based on the contents of the code base?


```python
!nsys profile --stats=true ./sm-optimized-vector-add
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    No such file or directory: ./sm-optimized-vector-add


### Exercise: Initialize Vector in Kernel

When `nsys profile` gives the amount of time that a kernel takes to execute, the host-to-device page faults and data migrations that occur during this kernel's execution are included in the displayed execution time.

With this in mind, refactor the `initWith` host function in your [01-vector-add.cu](../edit/01-vector-add/01-vector-add.cu) program to instead be a CUDA kernel, initializing the allocated vector in parallel on the GPU. After successfully compiling and running the refactored application, but before profiling it, hypothesize about the following:

- How do you expect the refactor to affect UM memory migration behavior?
- How do you expect the refactor to affect the reported run time of `addVectorsInto`?

Once again, record the results. Refer to [the solution](../edit/07-init-in-kernel/solutions/01-vector-add-init-in-kernel-solution.cu) if you get stuck.


```python
!nvcc -o initialize-in-kernel 01-vector-add/01-vector-add.cu -run
```

    Success! All values calculated correctly.



```python
!nsys profile --stats=true ./initialize-in-kernel
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Success! All values calculated correctly.
    Processing events...
    Saving temporary "/tmp/nsys-report-1f95-7837-8ef1-ad06.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-1f95-7837-8ef1-ad06.qdrep"
    Exporting 4261 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-1f95-7837-8ef1-ad06.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls    Average     Minimum    Maximum           Name         
     -------  ---------------  ---------  -----------  ---------  ---------  ---------------------
        52.9        246568090          3   82189363.3      17847  246515096  cudaMallocManaged    
        42.1        196348103          1  196348103.0  196348103  196348103  cudaDeviceSynchronize
         4.9         22875138          3    7625046.0    6820876    9174299  cudaFree             
         0.0            50862          1      50862.0      50862      50862  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances    Average     Minimum    Maximum                      Name                    
     -------  ---------------  ---------  -----------  ---------  ---------  -------------------------------------------
       100.0        196339850          1  196339850.0  196339850  196339850  addVectorsInto(float*, float*, float*, int)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
     -------  ---------------  ----------  -------  -------  -------  ---------------------------------
        76.7         69375771        2304  30111.0     2238   182590  [CUDA Unified Memory memcpy HtoD]
        23.3         21056850         768  27417.8     1503   159903  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total     Operations  Average  Minimum  Maximum               Operation            
     ----------  ----------  -------  -------  --------  ---------------------------------
     393216.000        2304  170.667    4.000  1020.000  [CUDA Unified Memory memcpy HtoD]
     131072.000         768  170.667    4.000  1020.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum              Name           
     -------  ---------------  ---------  ----------  -------  ---------  --------------------------
        83.9       1484251218         78  19028861.8    29910  100134112  poll                      
         8.8        155681479         68   2289433.5    18245   20781689  sem_timedwait             
         5.7        100650647        676    148891.5     1007   17688595  ioctl                     
         1.4         25372091         98    258898.9     1484    9115204  mmap                      
         0.1          1645325         82     20064.9     5864      44643  open64                    
         0.0           218316          3     72772.0    67740      79905  fgets                     
         0.0           136802          4     34200.5    33061      35812  pthread_create            
         0.0           126276         25      5051.0     1450      24129  fopen                     
         0.0           113706         11     10336.9     2941      25616  write                     
         0.0            42101         12      3508.4     1657       5386  munmap                    
         0.0            40969          3     13656.3    11187      16563  pthread_rwlock_timedwrlock
         0.0            29386         18      1632.6     1012       4263  fclose                    
         0.0            27202         20      1360.1     1016       4039  fcntl                     
         0.0            25512          5      5102.4     3133       7637  open                      
         0.0            23035         13      1771.9     1044       2983  read                      
         0.0            20350          6      3391.7     1000       9163  fgetc                     
         0.0             8863          2      4431.5     4194       4669  socket                    
         0.0             8806          3      2935.3     1480       3776  fread                     
         0.0             8129          1      8129.0     8129       8129  pipe2                     
         0.0             7609          4      1902.3     1818       2014  mprotect                  
         0.0             5470          1      5470.0     5470       5470  connect                   
         0.0             2342          1      2342.0     2342       2342  bind                      
         0.0             1450          1      1450.0     1450       1450  listen                    
    
    Report file moved to "/dli/task/report5.qdrep"
    Report file moved to "/dli/task/report5.sqlite"
    


---
## Asynchronous Memory Prefetching

A powerful technique to reduce the overhead of page faulting and on-demand memory migrations, both in host-to-device and device-to-host memory transfers, is called **asynchronous memory prefetching**. Using this technique allows programmers to asynchronously migrate unified memory (UM) to any CPU or GPU device in the system, in the background, prior to its use by application code. By doing this, GPU kernels and CPU function performance can be increased on account of reduced page fault and on-demand data migration overhead.

Prefetching also tends to migrate data in larger chunks, and therefore fewer trips, than on-demand migration. This makes it an excellent fit when data access needs are known before runtime, and when data access patterns are not sparse.

CUDA Makes asynchronously prefetching managed memory to either a GPU device or the CPU easy with its `cudaMemPrefetchAsync` function. Here is an example of using it to both prefetch data to the currently active GPU device, and then, to the CPU:

```cpp
int deviceId;
cudaGetDevice(&deviceId);                                         // The ID of the currently active GPU device.

cudaMemPrefetchAsync(pointerToSomeUMData, size, deviceId);        // Prefetch to GPU device.
cudaMemPrefetchAsync(pointerToSomeUMData, size, cudaCpuDeviceId); // Prefetch to host. `cudaCpuDeviceId` is a
                                                                  // built-in CUDA variable.
```

### Exercise: Prefetch Memory

At this point in the lab, your [01-vector-add.cu](../edit/01-vector-add/01-vector-add.cu) program should not only be launching a CUDA kernel to add 2 vectors into a third solution vector, all which are allocated with `cudaMallocManaged`, but should also be initializing each of the 3 vectors in parallel in a CUDA kernel. If for some reason, your application does not do any of the above, please refer to the following [reference application](../edit/07-init-in-kernel/solutions/01-vector-add-init-in-kernel-solution.cu), and update your own code bases to reflect its current functionality.

Conduct 3 experiments using `cudaMemPrefetchAsync` inside of your [01-vector-add.cu](../edit/01-vector-add/01-vector-add.cu) application to understand its impact on page-faulting and memory migration.

- What happens when you prefetch one of the initialized vectors to the device?
- What happens when you prefetch two of the initialized vectors to the device?
- What happens when you prefetch all three of the initialized vectors to the device?

Hypothesize about UM behavior, page faulting specifically, as well as the impact on the reported run time of the initialization kernel, before each experiment, and then verify by running `nsys profile`. Refer to [the solution](../edit/08-prefetch/solutions/01-vector-add-prefetch-solution.cu) if you get stuck.


```python
!nvcc -o prefetch-to-gpu 01-vector-add/01-vector-add.cu -run
```

    Success! All values calculated correctly.



```python
!nsys profile --stats=true ./prefetch-to-gpu
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Success! All values calculated correctly.
    Processing events...
    Saving temporary "/tmp/nsys-report-f6e7-8150-e726-67a7.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-f6e7-8150-e726-67a7.qdrep"
    Exporting 1324 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-f6e7-8150-e726-67a7.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum           Name         
     -------  ---------------  ---------  ----------  -------  ---------  ---------------------
        60.8        246929752          3  82309917.3    18809  246857569  cudaMallocManaged    
        32.0        130007173          6  21667862.2   645335   43686646  cudaMemPrefetchAsync 
         6.4         25905844          3   8635281.3  8320458    8873167  cudaFree             
         0.9          3565876          2   1782938.0     1844    3564032  cudaDeviceSynchronize
         0.0            48950          4     12237.5     4803      31700  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances   Average   Minimum  Maximum                     Name                    
     -------  ---------------  ---------  ---------  -------  -------  -------------------------------------------
        52.1          1863275          3   621091.7   618329   623833  initWith(float, float*, int)               
        47.9          1712077          1  1712077.0  1712077  1712077  addVectorsInto(float*, float*, float*, int)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average   Minimum  Maximum              Operation            
     -------  ---------------  ----------  --------  -------  -------  ---------------------------------
       100.0         61329579         192  319424.9   310941   339484  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total     Operations  Average   Minimum   Maximum               Operation            
     ----------  ----------  --------  --------  --------  ---------------------------------
     393216.000         192  2048.000  2048.000  2048.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum              Name           
     -------  ---------------  ---------  ----------  -------  ---------  --------------------------
        64.5        621971467         37  16810039.6    48396  100126989  poll                      
        23.8        229587670        682    336638.8     1035   43625762  ioctl                     
         8.5         81973683         32   2561677.6    14805   20806710  sem_timedwait             
         3.0         28481523         98    290627.8     1535    8706322  mmap                      
         0.2          1938904         82     23645.2     4571      41552  open64                    
         0.0           214738          3     71579.3    69725      74806  fgets                     
         0.0           157657          4     39414.3    33525      45096  pthread_create            
         0.0           131104         25      5244.2     1696      22634  fopen                     
         0.0           104527         11      9502.5     4366      14919  write                     
         0.0            44969         11      4088.1     1864       6211  munmap                    
         0.0            37847         22      1720.3     1061      12196  fcntl                     
         0.0            34473          7      4924.7     1031      11076  fgetc                     
         0.0            31036          5      6207.2     3232       9279  open                      
         0.0            27743         18      1541.3     1040       4276  fclose                    
         0.0            27681          3      9227.0     3833      19186  fread                     
         0.0            22452          2     11226.0    11038      11414  pthread_rwlock_timedwrlock
         0.0            22432         12      1869.3     1137       3090  read                      
         0.0            15838          2      7919.0     6764       9074  socket                    
         0.0             8758          1      8758.0     8758       8758  connect                   
         0.0             7832          1      7832.0     7832       7832  pipe2                     
         0.0             7592          4      1898.0     1647       2278  mprotect                  
         0.0             2647          1      2647.0     2647       2647  bind                      
         0.0             1825          1      1825.0     1825       1825  listen                    
    
    Report file moved to "/dli/task/report9.qdrep"
    Report file moved to "/dli/task/report9.sqlite"
    


### Exercise: Prefetch Memory Back to the CPU

Add additional prefetching back to the CPU for the function that verifies the correctness of the `addVectorInto` kernel. Again, hypothesize about the impact on UM before profiling in `nsys` to confirm. Refer to [the solution](../edit/08-prefetch/solutions/02-vector-add-prefetch-solution-cpu-also.cu) if you get stuck.


```python
!nvcc -o prefetch-to-cpu 01-vector-add/01-vector-add.cu -run
```

    Success! All values calculated correctly.



```python
!nsys profile --stats=true ./prefetch-to-cpu
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    Success! All values calculated correctly.
    Processing events...
    Saving temporary "/tmp/nsys-report-b015-827b-1d8e-94b5.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-b015-827b-1d8e-94b5.qdrep"
    Exporting 1869 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-b015-827b-1d8e-94b5.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum           Name         
     -------  ---------------  ---------  ----------  -------  ---------  ---------------------
        90.7        226355761          3  75451920.3    16901  226306390  cudaMallocManaged    
         7.0         17463835          3   5821278.3   827356   15560677  cudaFree             
         1.4          3568540          1   3568540.0  3568540    3568540  cudaDeviceSynchronize
         0.8          2047582          3    682527.3   660786     708274  cudaMemPrefetchAsync 
         0.0            45117          4     11279.3     5004      28373  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances   Average   Minimum  Maximum                     Name                    
     -------  ---------------  ---------  ---------  -------  -------  -------------------------------------------
        52.2          1869676          3   623225.3   623097   623386  initWith(float, float*, int)               
        47.8          1709966          1  1709966.0  1709966  1709966  addVectorsInto(float*, float*, float*, int)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
     -------  ---------------  ----------  -------  -------  -------  ---------------------------------
       100.0         21340665         768  27787.3     1662   178495  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total     Operations  Average  Minimum  Maximum               Operation            
     ----------  ----------  -------  -------  --------  ---------------------------------
     131072.000         768  170.667    4.000  1020.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum              Name           
     -------  ---------------  ---------  ----------  -------  ---------  --------------------------
        76.3        571482780         32  17858836.9    35573  100129513  poll                      
        11.5         85994736        679    126649.1     1013   17780615  ioctl                     
         9.3         69686655         27   2580987.2    19257   20805374  sem_timedwait             
         2.7         20087435         98    204973.8     1372   15505708  mmap                      
         0.2          1404635         82     17129.7     4452      29145  open64                    
         0.0           206913          3     68971.0    67678      71538  fgets                     
         0.0           141022          4     35255.5    33047      37923  pthread_create            
         0.0           112068         25      4482.7     1639      20610  fopen                     
         0.0            86027         11      7820.6     4368      11508  write                     
         0.0            54573          5     10914.6     7840      11889  pthread_rwlock_timedwrlock
         0.0            39317         12      3276.4     1649       5012  munmap                    
         0.0            27069         18      1503.8     1025       4105  fclose                    
         0.0            26120          5      5224.0     3052       7837  open                      
         0.0            24035         13      1848.8     1328       2990  read                      
         0.0            19613          5      3922.6     1043       9317  fgetc                     
         0.0            11233          1     11233.0    11233      11233  sem_wait                  
         0.0            10124          3      3374.7     1885       4606  fread                     
         0.0             9853          5      1970.6     1033       5409  fcntl                     
         0.0             9308          2      4654.0     4218       5090  socket                    
         0.0             7431          4      1857.8     1668       2020  mprotect                  
         0.0             6890          1      6890.0     6890       6890  pipe2                     
         0.0             5730          1      5730.0     5730       5730  connect                   
         0.0             2058          1      2058.0     2058       2058  bind                      
         0.0             1487          1      1487.0     1487       1487  listen                    
    
    Report file moved to "/dli/task/report10.qdrep"
    Report file moved to "/dli/task/report10.sqlite"
    


After this series of refactors to use asynchronous prefetching, you should see that there are fewer, but larger, memory transfers, and, that the kernel execution time is significantly decreased.

---
## Summary

At this point in the lab, you are able to:

- Use the Nsight Systems command line tool (**nsys**) to profile accelerated application performance.
- Leverage an understanding of **Streaming Multiprocessors** to optimize execution configurations.
- Understand the behavior of **Unified Memory** with regard to page faulting and data migrations.
- Use **asynchronous memory prefetching** to reduce page faults and data migrations for increased performance.
- Employ an iterative development cycle to rapidly accelerate and deploy applications.

In order to consolidate your learning, and reinforce your ability to iteratively accelerate, optimize, and deploy applications, please proceed to this lab's final exercise. After completing it, for those of you with time and interest, please proceed to the *Advanced Content* section.

---
## Final Exercise: Iteratively Optimize an Accelerated SAXPY Application

A basic accelerated SAXPY (Single Precision a\*x+b) application has been provided for you [here](../edit/09-saxpy/01-saxpy.cu). It currently contains a couple of bugs that you will need to find and fix before you can successfully compile, run, and then profile it with `nsys profile`.

After fixing the bugs and profiling the application, record the runtime of the `saxpy` kernel and then work *iteratively* to optimize the application, using `nsys profile` after each iteration to notice the effects of the code changes on kernel performance and UM behavior.

Utilize the techniques from this lab. To support your learning, utilize [effortful retrieval](http://sites.gsu.edu/scholarlyteaching/effortful-retrieval/) whenever possible, rather than rushing to look up the specifics of techniques from earlier in the lesson.

Your end goal is to profile an accurate `saxpy` kernel, without modifying `N`, to run in under *200us*. Check out [the solution](../edit/09-saxpy/solutions/02-saxpy-solution.cu) if you get stuck, and feel free to compile and profile it if you wish.


```python
!nvcc -o saxpy 09-saxpy/01-saxpy.cu -run
```

    c[0] = 5, c[1] = 0, c[2] = 0, c[3] = 0, c[4] = 0, 
    c[4194299] = 0, c[4194300] = 0, c[4194301] = 0, c[4194302] = 0, c[4194303] = 0, 



```python
!nsys profile --stats=true ./saxpy
```

    Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
    WARNING: The command line includes a target application therefore the CPU context-switch scope has been set to process-tree.
    Collecting data...
    c[0] = 5, c[1] = 0, c[2] = 0, c[3] = 0, c[4] = 0, 
    c[4194299] = 0, c[4194300] = 0, c[4194301] = 0, c[4194302] = 0, c[4194303] = 0, 
    Processing events...
    Saving temporary "/tmp/nsys-report-e226-e19f-46f0-dd2a.qdstrm" file to disk...
    
    Creating final output files...
    Processing [==============================================================100%]
    Saved report file to "/tmp/nsys-report-e226-e19f-46f0-dd2a.qdrep"
    Exporting 1128 events: [==================================================100%]
    
    Exported successfully to
    /tmp/nsys-report-e226-e19f-46f0-dd2a.sqlite
    
    
    CUDA API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum           Name         
     -------  ---------------  ---------  ----------  -------  ---------  ---------------------
        94.9        241602488          3  80534162.7    23578  241548215  cudaMallocManaged    
         3.4          8684580          1   8684580.0  8684580    8684580  cudaDeviceSynchronize
         1.0          2653643          3    884547.7   859968     914030  cudaFree             
         0.7          1681948          3    560649.3     7039    1501333  cudaMemPrefetchAsync 
         0.0            31025          1     31025.0    31025      31025  cudaLaunchKernel     
    
    
    
    CUDA Kernel Statistics:
    
     Time(%)  Total Time (ns)  Instances  Average   Minimum  Maximum           Name          
     -------  ---------------  ---------  --------  -------  -------  -----------------------
       100.0           244989          1  244989.0   244989   244989  saxpy(int*, int*, int*)
    
    
    
    CUDA Memory Operation Statistics (by time):
    
     Time(%)  Total Time (ns)  Operations  Average   Minimum  Maximum              Operation            
     -------  ---------------  ----------  --------  -------  -------  ---------------------------------
        99.7          8238147          24  343256.1   340028   354972  [CUDA Unified Memory memcpy HtoD]
         0.3            24670           4    6167.5     1727    10592  [CUDA Unified Memory memcpy DtoH]
    
    
    
    CUDA Memory Operation Statistics (by size in KiB):
    
       Total    Operations  Average   Minimum   Maximum               Operation            
     ---------  ----------  --------  --------  --------  ---------------------------------
     49152.000          24  2048.000  2048.000  2048.000  [CUDA Unified Memory memcpy HtoD]
       128.000           4    32.000     4.000    60.000  [CUDA Unified Memory memcpy DtoH]
    
    
    
    Operating System Runtime API Statistics:
    
     Time(%)  Total Time (ns)  Num Calls   Average    Minimum   Maximum              Name           
     -------  ---------------  ---------  ----------  -------  ---------  --------------------------
        70.8        381653688         26  14678988.0    47082  100124992  poll                      
        18.7        100680769        679    148278.0     1022   17786454  ioctl                     
         8.4         45427148         17   2672185.2    14894   20527213  sem_timedwait             
         1.0          5357067         98     54663.9     1445     834365  mmap                      
         0.6          3302933          3   1100977.7    25607    2000026  sem_wait                  
         0.4          1906435         82     23249.2     4866      41801  open64                    
         0.0           264655          5     52931.0    33187      95364  pthread_create            
         0.0           208343          3     69447.7    67566      73013  fgets                     
         0.0           128033         25      5121.3     1507      28755  fopen                     
         0.0           113524         13      8732.6     4324      13854  write                     
         0.0            76009         10      7600.9     1298      16699  fgetc                     
         0.0            46072          5      9214.4     1942      37080  mprotect                  
         0.0            36576         13      2813.5     1153      13536  read                      
         0.0            32765         23      1424.6     1000       4971  fcntl                     
         0.0            31188          9      3465.3     1821       5459  munmap                    
         0.0            30686          5      6137.2     3142       8539  open                      
         0.0            28950         18      1608.3     1025       3903  fclose                    
         0.0             9830          2      4915.0     4461       5369  socket                    
         0.0             8333          1      8333.0     8333       8333  pipe2                     
         0.0             7291          1      7291.0     7291       7291  pthread_rwlock_timedwrlock
         0.0             6866          2      3433.0     3430       3436  fread                     
         0.0             6611          1      6611.0     6611       6611  connect                   
         0.0             2355          1      2355.0     2355       2355  bind                      
         0.0             1587          1      1587.0     1587       1587  listen                    
    
    Report file moved to "/dli/task/report13.qdrep"
    Report file moved to "/dli/task/report13.sqlite"
    



```python

```
