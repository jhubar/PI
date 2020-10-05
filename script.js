$.getJSON("https://opendata.bruxelles.be/api/datasets/1.0/search/?q=covid",function(data){
  console.log(data);
  var cumulativeCases = datasets.2.fields.5.type;
  $('.type').append(cumulativeCases);
});
