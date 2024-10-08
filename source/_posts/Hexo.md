---
title: Hexo博客搭建与魔改汇总
date: 2024-09-19 21:21:44
tags: 
- Hexo
categories:
- Hexo
keywords:
- Hexo
cover: https://pic.imgdb.cn/item/66ed2ba7f21886ccc0ebc562.jpg
description: 起点
---
# 一切从这里开始

```py
print("Ciallo!")
```

{% span blue, 粗略完成了blog的搭建，说是搭建，事实上也只是站在前人的肩膀上按部就班地组装hexo这个框架。为了方便以后更好地维护和查找问题，在下面列出一些至今用到的各类连接。 %}
{% del 静态页面太折磨了Orz %}

## 搭建流程

{% folding cyan,点这里 %}
{% link Hexo+Next主题搭建个人博客+优化全过程（完整详细版）, https://zhuanlan.zhihu.com/p/618864711, %}
{% endfolding %}
原博主已经弃坑Hexo，改用Notion + vercel搭建博客了，不过留下的内容足够详细，可供参考。
## 客制化相关

### 配置自定义css与js文件

{% folding cyan,点这里 %}
{% link Hexo博客添加自定义css和js文件,https://blog.leonus.cn/2022/custom.html, %}
{% endfolding %}

### 首页文章滑动卡片布局

{% folding cyan,点这里 %}
{% link 双栏布局首页卡片魔改教程,https://akilar.top/posts/d6b69c49/, %}
{% endfolding %}

### 外挂标签PLUS

{% folding cyan,点这里 %}
{% link Tag Plugins Plus,https://akilar.top/posts/615e2dec/, %}
{% endfolding %}

{% folding cyan,注意事项 %}
{% tip bell %}
使用此教程中的链接卡片时，教程中并没有说如何添加默认图片。经网页审查元素发现，头图地址默认指向 {% span red, https://你的用户名.github.io/img/link.png %}
如果不想给每个链接都添加一个图片地址，那么可以把图片链接留空，并在 {% span red, 根目录/soure %} 下新建一个img文件夹，在文件夹中放入你想要的默认图片，并重命名为 {% span red, link.png %} 即可
{% endtip %}
{% endfolding %}

### 时间轴添加对应生肖

{% folding cyan,点这里 %}
{% link Archive Beautify,https://akilar.top/posts/22257072/, %}
{% endfolding %}

### 入站Loading动画

{% folding cyan,点这里 %}
{% link Heo同款loading动画,https://blog.anheyu.com/posts/52d8.html, %}
{% endfolding %}

### 页面切换时更改网站标题

{% folding cyan,点这里 %}
{% link 网站恶搞标题,https://www.fomal.cc/posts/d1927166.html#%E7%BD%91%E7%AB%99%E6%81%B6%E6%90%9E%E6%A0%87%E9%A2%98, %}
{% endfolding %}

```py
//动态标题
var OriginTitile = document.title;
var titleTime;
document.addEventListener("visibilitychange", function () {
  if (document.hidden) {
    //离开当前页面时标签显示内容
    document.title = "再会了，雷狼龙";
    clearTimeout(titleTime);
  } else {
    //返回当前页面时标签显示内容
    document.title = "我即优雅又充满良心";
    //四秒后变回正常标题
    titleTime = setTimeout(function () {
      document.title = OriginTitile;
    }, 4000);
  }
});
```


