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


var percentTOTALRECOVERED = 19679/130235;
var percentTOTALDeaths = 10064/130235;
var percentTotalConfirmed = (130235-percentTOTALRECOVERED-percentTOTALDeaths)/130235;
var myPieChart = new Chart(ctx, {
  type: 'doughnut',
  data: {
    labels: ["TotalRecovered", "TotalDeaths", "TotalConfirmed"],
    datasets: [{
      data: [percentTOTALRECOVERED, percentTOTALDeaths, percentTotalConfirmed],
      backgroundColor: ['#4e73df', '#1cc88a', '#36b9cc'],
      hoverBackgroundColor: ['#2e59d9', '#17a673', '#2c9faf'],
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
