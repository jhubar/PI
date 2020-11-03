// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';
const $url = "https://raw.githubusercontent.com/julien1941/PI/master/Python/Data/SEIR.json?token=AL3RLGMAO5MHKOLL2O7CVLC7VLF6U"



loadData();

function loadData(){

    var data = ''
    // DAp
    var tmp ;

    $.get($url,function(data){

      const result = JSON.parse(data);


      data_day_fit = [];
      data_seir_fit_cumul_pos = [];
      data_seir_fit_hospit = [];
      data_seir_fit_cumul_pos_fit = [];
      data_seir_fit_hospit_fit = [];



      for(var i=0;i<result.log.length-1;i++){

        data_day_fit.push(result.log[i].day);
        data_seir_fit_cumul_pos.push(result.log[i].cumul_positive);
        data_seir_fit_hospit.push(result.log[i].hospit);
        data_seir_fit_cumul_pos_fit.push(result.log[i].cumul_positive_fit);
        data_seir_fit_hospit_fit.push(result.log[i].hospit_fit);


      }



  var ctx_active_cases = document.getElementById("myAreaSeirfit");


  draw();


},
);}

function number_format(number, decimals, dec_point, thousands_sep) {
  // *     example: number_format(1234.56, 2, ',', ' ');
  // *     return: '1 234,56'
  number = (number + '').replace(',', '').replace(' ', '');
  var n = !isFinite(+number) ? 0 : +number,
    prec = !isFinite(+decimals) ? 0 : Math.abs(decimals),
    sep = (typeof thousands_sep === 'undefined') ? ',' : thousands_sep,
    dec = (typeof dec_point === 'undefined') ? '.' : dec_point,
    s = '',
    toFixedFix = function(n, prec) {
      var k = Math.pow(10, prec);
      return '' + Math.round(n * k) / k;
    };
  // Fix for IE parseFloat(0.55).toFixed(0) = 0;
  s = (prec ? toFixedFix(n, prec) : '' + Math.round(n)).split('.');
  if (s[0].length > 3) {
    s[0] = s[0].replace(/\B(?=(?:\d{3})+(?!\d))/g, sep);
  }
  if ((s[1] || '').length < prec) {
    s[1] = s[1] || '';
    s[1] += new Array(prec - s[1].length + 1).join('0');
  }
  return s.join(dec);
}







function draw(ans) {

  if (typeof(myLineChart) != "undefined"){

    myLineChart.destroy();

  }

  ctx_active_cases = document.getElementById("myAreaSeirModel");
  myLineChart = new Chart(ctx_active_cases, {
    type: 'line',
    data: {
      labels: data_day_fit,
      datasets: [
        // Susceptible
        {
          label: "Susceptible ",
          lineTension: 0.6,
          backgroundColor: "rgba(78, 115, 223, 0.2)",
          borderColor: "rgba(78, 115, 223, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(78, 115, 223, 1)",
          pointBorderColor: "rgba(78, 115, 223, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",
          pointHoverBorderColor: "rgba(78, 115, 223, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: data_seir_fit_hospit_fit,
        },
        // Exposed
        {
          label: "Exposed ",
          lineTension: 0.6,
          backgroundColor: "rgba(240, 173, 78, 0.2)",
          borderColor: "rgba(240, 173, 78, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(240, 173, 78, 1)",
          pointBorderColor: "rgba(240, 173, 78, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(240, 173, 78, 1)",
          pointHoverBorderColor: "rgba(240, 173, 78, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: data_seir_fit_hospit,
        },
        // Infectious
        {
          label: "Infectious ",
          lineTension: 0.3,
          backgroundColor: "rgba(237, 0, 59,0.1)",
          borderColor: "rgba(237, 0, 59, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: data_seir_fit_cumul_pos,
        },
        // Recovered
        {
          label: "Recovered ",
          lineTension: 0.6,
          backgroundColor: "rgba(37, 56, 60, 0.2)",
          borderColor: "rgba(37, 56, 60, 0.1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(37, 56, 60, 0.1)",
          pointBorderColor: "rgba(37, 56, 60, 0.1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(37, 56, 60, 0.1)",
          pointHoverBorderColor: "rgba(37, 56, 60, 0.1)",
          pointHitRadius: 5,
          pointBorderWidth: 1,
          data: data_seir_fit_cumul_pos_fit,
        },






    ],
    },
    options: {
      maintainAspectRatio: false,
      layout: {
        padding: {
          left: 10,
          right: 25,
          top: 25,
          bottom: 0
        }
      },
      scales: {
        xAxes: [{
          time: {
            unit: 'date'
          },
          gridLines: {
            display: false,
            drawBorder: false
          },
          ticks: {
            maxTicksLimit: 7,
            // callback: function(value, index, values) {
            //   return number_format(value)+ ' Days';
            // }
          }
        }],
        yAxes: [{
          ticks: {
            maxTicksLimit: 5,
            padding: 10,

            callback: function(value, index, values) {
              return number_format(value)+ ' Cases';
            }
          },
          gridLines: {
            color: "rgb(234, 236, 244)",
            zeroLineColor: "rgb(234, 236, 244)",
            drawBorder: false,
            borderDash: [2],
            zeroLineBorderDash: [2]
          }
        }],
      },
      legend: {
        display: false
      },
      tooltips: {
        backgroundColor: "rgb(255,255,255)",
        bodyFontColor: "#858796",
        titleMarginBottom: 10,
        titleFontColor: '#6e707e',
        titleFontSize: 14,
        borderColor: '#dddfeb',
        borderWidth: 1,
        xPadding: 15,
        yPadding: 15,
        displayColors: false,
        intersect: false,
        mode: 'index',
        caretPadding: 10,
        callbacks: {
          label: function(tooltipItem, chart) {
            var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
            return datasetLabel + number_format(tooltipItem.yLabel)+ ': Cases';
          }
        }
      }
    }
  },);


}
