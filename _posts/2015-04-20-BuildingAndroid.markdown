---
layout: post
title:  Building Android on Mac
date:   2015-04-20 13:09:39
categories: android
tags:
permalink: /:categories/:title.html
published: true
---

Recently I needed to build the Android Open Source Project (AOSP). The build instructions on the [Android's website] is helpful but I had problems getting the correct JDK and Xcode command line tools as each [branch] requires a different version. After some digging I figured out how to set the required versions. Follow the initial build instructions and this tutorial will show you how to setup the appropriate versions.

## Java
Download and install the appropriate java version (you don't need to uninstall your current version). Open up a terminal and run this command ```/usr/libexec/java_home -V```. You should get something similar to the following:

{% highlight bash %}
Matching Java Virtual Machines (4):
    1.8.0_11, x86_64: "Java SE 8" /Library/Java/JavaVirtualMachines/jdk1.8.0_11.jdk/Contents/Home
    1.7.0_71, x86_64: "Java SE 7" /Library/Java/JavaVirtualMachines/jdk1.7.0_71.jdk/Contents/Home
    1.6.0_65-b14-466.1, x86_64: "Java SE 6" /System/Library/Java/JavaVirtualMachines/1.6.0.jdk/Contents/Home
{% endhighlight %}

Make note of the java version you need and add this to your .bash_profile. For example, if I need java version 1.7 I would put the following in my .bash_profile.

{% highlight bash %}
export JAVA_HOME=`/usr/libexec/java_home -v 1.7.0_71`
{% endhighlight %}


## Xcode
Next download the appropriate Xcode package from the [Apple developers website]. Once you download this, simply mount the volume. Open up Xcode -> Preferences -> Locations. Under command line tools you should be able to select the downloaded version of the command line tools as in the picture below. (This can also be done via xcode-select)

![Xcode select]({{ site.url }}/images/xcode_select.png)


Build away....



[link]:   http://link
[Android's website]: https://source.android.com/source/initializing.html
[branch]: https://source.android.com/source/initializing.html#master-branch
[Apple developers website]: https://developer.apple.com/downloads/index.action
