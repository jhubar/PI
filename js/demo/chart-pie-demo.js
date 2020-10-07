// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

// Pie Chart Example
var ctx = document.getElementById("myPieChart");
// var percentTOTALRECOVERED = 19679/13023;
function retrivedata(){
  var url = "https://api.covid19api.com/summary "
  var data = ''
  $.get(url,function(data){
      return data.Countries[16].TotalConfirmed

  })
}
function loadData(){
  var url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
  var data = ''
  // DAp
  var tmp ;
  $.get(url,function(data){
    var result = [];
    var lines=data.split("\n");
    var headers=lines[0].split(",");
    for(var i=1;i<lines.length;i++){
      var obj = {};
      var currentline=lines[i].split(",");
      for(var j=0;j<headers.length;j++){
        obj[headers[j]] = currentline[j];
      }
      result.push(obj);
    }

    var size = result.length-1;
    label = [];
    dataL = [];
    for(var i=0;i<result.length;i++){
      label.push(result[i].Day);
      dataL.push(result[i].num_positive);
    }
    var num_positive = result[size-1].num_positive
    var numberOfHospitalied = result[size-1].num_hospitalised/num_positive;
    var fatalities = result[size-1].num_fatalities/num_positive;
    var critical = result[size-1].num_critical/num_positive;
    var normalCases = (num_positive-numberOfHospitalied-fatalities)/num_positive;

    var ctx = document.getElementById("myPieChart");
    var myPieChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ["normal Cases", "number Of Hospitalied", "critical","fatalities"],
        datasets: [{
          data: [normalCases,numberOfHospitalied, critical, fatalities],
          backgroundColor: ['#4e73df','#1cc88a', '#ffc107','#DF2C2C'],
          hoverBackgroundColor: ['#2e59d9', '#17a673', '#2c9faf','#DF2C2C'],
          hoverBorderColor: "rgba(234, 236, 244, 1)",
        }],
      },
      options: {
        maintainAspectRatio: false,
        tooltips: {
          backgroundColor: "rgb(255,255,255)",
          bodyFontColor: "#858796",
          borderColor: '#dddfeb',
          borderWidth: 1,
          xPadding: 15,
          yPadding: 15,
          displayColors: false,
          caretPadding: 10,
        },
        legend: {
          display: false
        },
        cutoutPercentage: 80,
      },
    });
  });
}

var numberOfHospitalied = 0/5;
var fatalities = 0/5;
var critical = 0/5;
var normalCases = (5-numberOfHospitalied-fatalities)/5;
var myPieChart = new Chart(ctx, {
  type: 'doughnut',
  data: {
    labels: ["normal Cases", "number Of Hospitalied", "critical","fatalities"],
    datasets: [{
      data: [normalCases,numberOfHospitalied, critical, fatalities],
      backgroundColor: ['#1cc88a','#4e73df', '#ffc107','#DF2C2C'],
      hoverBackgroundColor: ['#2e59d9', '#17a673', '#2c9faf','#DF2C2C'],
      hoverBorderColor: "rgba(234, 236, 244, 1)",
    }],
  },
  options: {
    maintainAspectRatio: false,
    tooltips: {
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: false,
      caretPadding: 10,
    },
    legend: {
      display: false
    },
    cutoutPercentage: 80,
  },
});
loadData();
