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