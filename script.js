
function init(){
  var url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
  var data = ''
  $.get(url,function(data){
    console.log(data)
    function csvJSON(csv){
      var lines=csv.split("\n");
      var result = [];
      var headers=lines[0].split(",");
      for(var i=1;i<lines.length;i++){
        var obj = {};
        var currentline=lines[i].split(",");
        for(var j=0;j<headers.length;j++){
          obj[headers[j]] = currentline[j];
        }
        result.push(obj);
      }
      //return result; //JavaScript object
      $("#result").html(result)
      // return JSON.stringify(result); //JSON

  })
}

init()
