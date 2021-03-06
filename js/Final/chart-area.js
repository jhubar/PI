// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';
const $url_data_prof = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/Cov_invaders.csv"
const $url_data_day_graph = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/days.csv"
const $url_data_day_wm = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/days0.csv"
const $url_data_day_sd = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/days1.csv"
const $url_data_day_hq = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/days2.csv"
const $url_data_day_cs = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/days3.csv"
const $url_data_day_ci = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/days4.csv"
const $url_data_day_ld = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/days5.csv"



// const $value_time_period_data = $('.value_time_period_data');
// const $value_time_data = $('#range_time_period_data');

const $value_time_SEIR = $('#range_time_period_SEIR');
const $value_time_period_SEIR = $('.value_time_period_SEIR');


const $value_time_wm = $('#range_time_period_wm');
const $value_time_period_wm = $('.range_time_wm');

const $value_time_sd = $('#range_time_period_sd');
const $value_time_period_sd = $('.range_time_sd');

const $value_time_hq = $('#range_time_period_hq');
const $value_time_period_hq = $('.range_time_hq');

const $value_time_cs = $('#range_time_period_cs');
const $value_time_period_cs = $('.range_time_cs');

const $value_time_ci = $('#range_time_period_ci');
const $value_time_period_ci = $('.range_time_ci');

const $value_time_ld = $('#range_time_period_ld');
const $value_time_period_ld = $('.range_time_ld');

// $value_time_period_data.html($value_time_data.val());
$value_time_period_SEIR.html($value_time_SEIR.val());
$value_time_period_wm.html($value_time_wm.val());
$value_time_period_sd.html($value_time_sd.val());
$value_time_period_hq.html($value_time_hq.val());
$value_time_period_cs.html($value_time_cs.val());
$value_time_period_ci.html($value_time_ci.val());
$value_time_period_ld.html($value_time_ld.val());

const $id_switch_positive = document.getElementById('customSwitches_Positive');
const $id_switch_hospitalized = document.getElementById('customSwitches_hospitalized');
const $id_switch_cum_hospitalized = document.getElementById('customSwitches_cum_hospitalized');
const $id_switch_target_data = document.getElementById('customSwitches_target_data');
// const $id_switch_criticals = document.getElementById('customSwitches_criticals');
// const $id_switch_fatalies = document.getElementById('customSwitches_fatalities');

const $value_customSwitchesSusceptible = $('#customSwitchesSusceptible');

const $id_switch_Susceptible = document.getElementById('customSwitchesSusceptible');
const $id_switch_Exosed = document.getElementById('customSwitchesExposed');
const $id_switch_Infected = document.getElementById('customSwitchesInfectious');
const $id_switch_Recovered = document.getElementById('customSwitchesRecovered');
const $id_switch_Hospitalized = document.getElementById('customSwitchesHospitalized');
const $id_switch_Criticals = document.getElementById('customSwitchesSeirCriticales');
const $id_switch_Death = document.getElementById('customSwitchesSeirFatalities');
const $id_switch_hq = document.getElementById('customSwitches_hq');
const $id_switch_ci = document.getElementById('customSwitches_ci');

// const $id_switch_num_bed_hos = document.getElementById('customSwitches_num_bed_hos');
// const $id_switch_num_bed_icu = document.getElementById('customSwitches_num_bed_icu');
$id_switch_hq.addEventListener('change',function(){
    load_day_constraint()
    draw();
});
$id_switch_ci.addEventListener('change',function(){
    load_day_constraint()
    draw();
});

$id_switch_target_data.addEventListener('change',function(){
    draw();
});

$id_switch_Susceptible.addEventListener('change',function(){
    draw();
});

$id_switch_Exosed.addEventListener('change',function(){
    draw();
});

$id_switch_Infected.addEventListener('change',function(){
    draw();
});

$id_switch_Recovered.addEventListener('change',function(){
    draw();
});

$id_switch_Hospitalized.addEventListener('change',function(){
    draw();
});

$id_switch_Criticals.addEventListener('change',function(){
    draw();
});

$id_switch_Death.addEventListener('change',function(){
    draw();
});
data_day =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299]
load_day_constraint();
loadData_prof();

load_day_wm();
load_day_sd();
load_day_cs();




function load_day_wm(){
  var data = ''
  // DAp
  var tmp ;

  $.get($url_data_day_wm,function(data){
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

    data_day_wm = []

    for(var i=0;i<$value_time_wm.val();i++){

      data_day_wm.push(result[i].Day);


    }

    var ctx_active_cases = document.getElementById("myAreaSeirModel");


    load_card_value_seir();
    load_day_constraint()
    draw()
},
);
}
function load_day_sd(){
  var data = ''
  // DAp
  var tmp ;

  $.get($url_data_day_sd,function(data){
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


    data_day_sd = []

    for(var i=0;i<$value_time_sd.val();i++){

      data_day_sd.push(result[i].Day);


    }

    var ctx_active_cases = document.getElementById("myAreaSeirModel");

    draw()
    load_card_value_seir();


},
);
}
function load_day_cs(){
  var data = ''
  // DAp
  var tmp ;

  $.get($url_data_day_cs,function(data){
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

    data_day_cs = []

    for(var i=0;i<$value_time_hq.val();i++){

      data_day_cs.push(result[i].Day);

    }

    var ctx_active_cases = document.getElementById("myAreaSeirModel");

    draw()
    load_card_value_seir();

},
);
}
function load_day_constraint(){
  var data = ''
  // DAp
  var tmp ;

  $.get($url_data_day_hq,function(data){
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

    data_day_hq = []

    for(var i=0;i<$value_time_hq.val();i++){

      data_day_hq.push(result[i].Day);


    }
    // constraint wm sd cs

    if(parseInt($value_time_wm.val()) == 1) {
      if (parseInt($value_time_sd.val()) == 1) {
        if (parseInt($value_time_cs.val()) == 1) {

          if ($id_switch_hq.checked == false && $id_switch_ci.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_0.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            console.log("ci false hq true")
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_0.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            console.log("ci true hq false")
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_0.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            console.log("ci true hq true")
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_0.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_1.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_1.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_1.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_1.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_2.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_2.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_2.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_2.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_3.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_3.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_3.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_3.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_4.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_4.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_4.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_4.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 2) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_5.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_5.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_5.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_5.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_6.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_6.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_6.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_6.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_7.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_7.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_7.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_7.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_8.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_8.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_8.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_8.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_9.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_9.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_9.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_9.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 3) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_10.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_10.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_10.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_10.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_11.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_11.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_11.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_11.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_12.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_12.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_12.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_12.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_13.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_13.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_13.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_13.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_14.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_14.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_14.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_14.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 4) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_15.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_15.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_15.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_15.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_16.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_16.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_16.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_16.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_17.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_17.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_17.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_17.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_18.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_18.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_18.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_18.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_19.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_19.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_19.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_19.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 5) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_20.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_20.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_20.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_20.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_21.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_21.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_21.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_21.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_22.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_22.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_22.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_22.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_23.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_23.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_23.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_23.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_24.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_24.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_24.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_24.csv"
          }
         loadData_scenario()
        }
      }
    }
    else if(parseInt($value_time_wm.val()) == 2) {
      if (parseInt($value_time_sd.val()) == 1) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_25.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_25.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_25.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_25.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_26.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_26.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_26.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_26.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_27.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_27.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_27.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_27.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_28.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_28.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_28.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_28.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_29.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_29.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_29.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_29.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 2) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_30.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_30.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_30.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_30.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_31.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_31.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_31.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_31.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_32.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_32.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_32.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_32.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_33.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_33.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_33.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_33.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_34.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_34.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_34.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_34.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 3) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_35.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_35.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_35.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_35.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_36.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_36.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_36.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_36.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_37.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_37.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_37.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_37.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_39.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_38.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_38.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_38.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_39.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_39.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_39.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_39.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 4) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_40.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_40.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_40.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_40.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_41.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_41.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_41.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_41.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_42.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_42.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_42.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_42.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_43.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_43.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_43.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_43.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_44.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_44.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_44.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_44.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 5) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_45.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_45.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_45.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_45.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_46.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_46.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_46.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_46.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_47.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_47.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_47.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_47.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_48.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_48.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_48.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_48.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_49.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_49.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_49.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_49.csv"
          }
         loadData_scenario()
        }
      }
    }
    else if(parseInt($value_time_wm.val()) == 3) {
      if (parseInt($value_time_sd.val()) == 1) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_50.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_50.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_50.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_50.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_51.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_51.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_51.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_51.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_52.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_52.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_52.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_52.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_53.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_53.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_53.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_53.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_54.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_54.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_54.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_54.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 2) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_55.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_55.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_55.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_55.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_56.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_56.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_56.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_56.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_57.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_57.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_57.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_57.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_58.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_58.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_58.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_58.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_59.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_59.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_59.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_59.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 3) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_60.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_60.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_60.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_60.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_61.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_61.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_61.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_61.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_62.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_62.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_62.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_62.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_63.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_63.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_63.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_63.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_64.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_64.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_64.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_64.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 4) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_65.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_65.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_65.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_65.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_66.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_66.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_66.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_66.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_67.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_67.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_67.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_67.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_68.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_68.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_68.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_68.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_69.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_69.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_69.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_69.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 5) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_70.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_70.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_70.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_70.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_71.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_71.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_71.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_71.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_72.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_72.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_72.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_72.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_73.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_73.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_73.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_73.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_74.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_74.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_74.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_74.csv"
          }
         loadData_scenario()
        }
      }
    }
    else if(parseInt($value_time_wm.val()) == 4) {
      if (parseInt($value_time_sd.val()) == 1) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_75.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_75.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_75.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_75.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_76.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_76csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_76.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_76.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_77.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_77.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_77.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_77.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_78.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_78.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_78.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_78.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_79.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_79.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_79.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_79.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 2) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_80.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_80.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_80.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_80.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_81.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_81.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_81.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_81.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_82.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_82.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_82.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_82.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_83.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_83.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_83.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_83.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_84.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_84.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_84.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_84.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 3) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_85.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_85.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_85.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_85.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_86.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_86.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_86.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_86.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_87.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_87.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_87.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_87.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_88.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_88.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_88.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_88.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_89.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_89.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_89.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_89.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 4) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_90.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_90.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_90.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_90.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_91.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_91.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_91.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_91.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_92.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_92.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_92.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_92.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_93.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_93.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_93.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_93.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_94.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_94.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_94.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_94.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 5) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_95.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_95.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_95.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_95.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_96.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_96.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_96.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_96.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_97.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_97.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_97.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_97.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_98.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_98.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_98.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_98.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_99.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_99.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_99.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_99.csv"
          }
         loadData_scenario()
        }
      }
    }
    else if(parseInt($value_time_wm.val()) == 5) {
      if (parseInt($value_time_sd.val()) == 1) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_100.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_100.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_100.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_100.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_101.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_101.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_101.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_101.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_102.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_102.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_102.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_102.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_103.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_103.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_103.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_103.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_104.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_104.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_104.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_104.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 2) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_105.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_105.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_105.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_105.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_106.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_106.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_106.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_106.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_107.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_107.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_107.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_107.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_108.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_108.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_108.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_108.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_109.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_109.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_109.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_109.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 3) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_110.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_110.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_110.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_110.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_111.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_111.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_111.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_111.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_112.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_112.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_112.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_112.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_113.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_113.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_113.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_113.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_114.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_114.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_114.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_114.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 4) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_115.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_115.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_115.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_115.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_116.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_116.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_116.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_116.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_117.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_117.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_117.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_117.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_118.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_118.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_118.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_118.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_119.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_119.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_119.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_119.csv"
          }
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 5) {
        if (parseInt($value_time_cs.val()) == 1) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_120.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_120.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_120.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_120.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 2) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_121.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_121.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_121.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_121.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 3) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_122.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_122.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_122.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_122.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 4) {
          if ($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_123.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_123.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_123.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_123.csv"
          }
         loadData_scenario()
        }
        else if (parseInt($value_time_cs.val()) == 5) {
          if ($id_switch_ci.checked == false && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_124.csv"
          }
          else if($id_switch_ci.checked == false && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_124.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == false){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ci_124.csv"
          }
          else if($id_switch_ci.checked == true && $id_switch_hq.checked == true){
            $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_hm_ci_124.csv"
          }
         loadData_scenario()
        }
      }
    }
    // constraint wm sd ld
    if(parseInt($value_time_wm.val()) == 1) {
      if (parseInt($value_time_sd.val()) == 1) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_0.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_1.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_2.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_3.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_4.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 2) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_5.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_6.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_7.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_8.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_9.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 3) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_10.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_11.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_12.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_13.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_14.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 4) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_15.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_16.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_17.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_18.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_19.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 5) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_20.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_21.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_22.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_23.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_24.csv"
         loadData_scenario()
        }
      }
    }
    else if(parseInt($value_time_wm.val()) == 2) {
      if (parseInt($value_time_sd.val()) == 1) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_25.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_26.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_27.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_28.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_29.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 2) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_30.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_31.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_32.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_33.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_34.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 3) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_35.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_36.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_37.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_38.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_39.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 4) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_40.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_41.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_42.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_43.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_44.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 5) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_45.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_46.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_47.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_48.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_49.csv"
         loadData_scenario()
        }
      }
    }
    else if(parseInt($value_time_wm.val()) == 3) {
      if (parseInt($value_time_sd.val()) == 1) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_50.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_51.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_52.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_53.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_54.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 2) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_55.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_56.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_57.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_58.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_59.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 3) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_60.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_61.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_62.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_63.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_64.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 4) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_65.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_66.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_67.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_68.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_69.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 5) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_70.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_71.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_72.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_73.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_74.csv"
         loadData_scenario()
        }
      }
    }
    else if(parseInt($value_time_wm.val()) == 4) {
      if (parseInt($value_time_sd.val()) == 1) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_75.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_76.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_77.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_78.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_79.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 2) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_80.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_81.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_82.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_83.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_84.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 3) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_85.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_86.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_87.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_88.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_89.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 4) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_90.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_91.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_92.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_93.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_94.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 5) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_95.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_96.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_97.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_98.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_99.csv"
         loadData_scenario()
        }
      }
    }
    else if(parseInt($value_time_wm.val()) == 5) {
      if (parseInt($value_time_sd.val()) == 1) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_100.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_101.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_102.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_103.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_104.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 2) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_105.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_106.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_107.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_108.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_109.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 3) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_110.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_111.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_112.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_113.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_114.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 4) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_115.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_116.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_117.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_118.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_119.csv"
         loadData_scenario()
        }
      }
      else if (parseInt($value_time_sd.val()) == 5) {
        if (parseInt($value_time_ld.val()) == 1) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_120.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 2) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_121.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 3) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_122.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 4) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_123.csv"
         loadData_scenario()
        }
        else if (parseInt($value_time_ld.val()) == 5) {
         $url_data_scenario = "https://raw.githubusercontent.com/jhubar/PI/master/BruteForceModel_V2/Data_Scenario/scenario_ld_124.csv"
         loadData_scenario()
        }
      }
    }
    draw()

},
);
}

function load_day_ci(){
  var data = ''
  // DAp
  var tmp ;

  $.get($url_data_day_ci,function(data){
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

    data_day_ci = []

    for(var i=0;i<$value_time_ci.val();i++){

      data_day_ci.push(result[i].Day);


    }

},
);
}
function load_day_ld(){
  var data = ''
  // DAp
  var tmp ;

  $.get($url_data_day_ld,function(data){
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

    data_day_ld = []

    for(var i=0;i<$value_time_ld.val();i++){

      data_day_ld.push(result[i].Day);


    }

},
);}

function loadData_scenario(){

    var data = ''
    // DAp
    var tmp ;

    $.get($url_data_scenario,function(data){
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


      data_day = []
      data_S = []
      data_E = []
      data_I = []
      data_R = []
      data_H = []
      data_C = []
      data_D = []



      for(var i=0;i<$value_time_SEIR.val();i++){
        data_day.push(i);
        data_S.push(result[i].S);
        data_E.push(result[i].E);
        data_I.push(result[i].I);
        data_R.push(result[i].R);
        data_H.push(result[i].H)
        data_C.push(result[i].C)
        data_D.push(result[i].D)


      }
  var ctx_active_cases = document.getElementById("myAreaSeirModel");

  draw();
},
);}
function loadData_prof(){

    var data = ''
    // DAp
    var tmp ;

    $.get($url_data_prof,function(data){
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



      data_prof_num_positive = []
      data_prof_num_tested = []
      data_prof_num_hospitalised = []
      data_prof_num_cumulative_hospitalizations = []
      data_prof_num_critical = []
      data_prof_num_fatalities = []
      data_ladder = []




      for(var i=0;i<$value_time_SEIR.val();i++){
        data_ladder.push(1500)

        if (i > 191){

          // data_prof_num_positive.push(0);
          // data_prof_num_tested.push(0);
          // data_prof_num_hospitalised.push(0);
          // data_prof_num_cumulative_hospitalizations.push(0);
          // data_prof_num_critical.push(0);
          // data_prof_num_fatalities.push(0);
        }
        else{

          data_prof_num_positive.push(result[i].num_positive);
          data_prof_num_tested.push(result[i].num_tested);
          data_prof_num_hospitalised.push(result[i].num_hospitalised);
          data_prof_num_cumulative_hospitalizations.push(result[i].num_cumulative_hospitalizations);
          data_prof_num_critical.push(result[i].num_critical);
          data_prof_num_fatalities.push(result[i].num_fatalities);
        }


      }




      var ctx_active_cases = document.getElementById("myAreaSeirModel");



      draw();






},
);}



$value_time_SEIR.on('input change', () => {

  $value_time_period_SEIR.html($value_time_SEIR.val());
  load_day_wm();
  load_day_sd();
  load_day_cs();
  load_day_constraint();
  loadData_prof();



});
// wm constraint
$value_time_wm.on('input change', () => {

  $value_time_period_wm.html($value_time_wm.val());
  load_day_wm();
  load_day_sd();
  load_day_cs();
  load_day_constraint();
  loadData_prof();
});
// sd constraint
$value_time_sd.on('input change', () => {

  $value_time_period_sd.html($value_time_sd.val());
  load_day_wm();
  load_day_sd();
  load_day_cs();
  load_day_constraint();

  loadData_prof();

});
// hq constraint
$value_time_hq.on('input change', () => {

  $value_time_period_hq.html($value_time_hq.val());
  load_day_wm();
  load_day_sd();
  load_day_cs();
  load_day_constraint();

  loadData_prof();
});
// CS constraint
$value_time_cs.on('input change', () => {

  $value_time_period_cs.html($value_time_cs.val());
  load_day_wm();
  load_day_sd();
  load_day_cs();


  if($value_time_cs.val() != 1){
    $value_time_ld.val(0)

  }
  load_day_constraint();
  load_day_ld();
  loadData_prof();

});
// Ci constraint
$value_time_ci.on('input change', () => {

  $value_time_period_ci.html($value_time_ci.val());
  load_day_wm();
  load_day_sd();
  load_day_cs();
  load_day_constraint();
  loadData_prof();


});
$value_time_ld.on('input change', () => {




  $value_time_period_ld.html($value_time_ld.val());
  load_day_wm();
  load_day_sd();

  if($value_time_ld.val() != 1){
    $value_time_cs.val(0)
  }
  load_day_cs();
  load_day_constraint();
  loadData_prof();


});

$("#num_Of_day_wm").html((($value_time_wm.val()-1)*30).toString())
$("#num_Of_day_sd").html((($value_time_sd.val()-1)*30).toString())
$("#num_Of_day_cs").html((($value_time_cs.val()-1)*30).toString())
$("#num_Of_day_ld").html((($value_time_ld.val()-1)*30).toString())

function load_card_value_seir(){
  $("#num_Of_Susceptible_seir").html((parseFloat(data_S[data_S.length-1]).toFixed(2)).toString())
  $("#num_Of_Exposed_seir").html((parseFloat(data_E[data_E.length-1]).toFixed(2)).toString())
  $("#num_Of_infected_seir").html((parseFloat(data_I[data_I.length-1]).toFixed(2)).toString())
  $("#num_Of_Recovered_seir").html((parseFloat(data_R[data_R.length-1]).toFixed(2)).toString())
  $("#num_Of_hospitalized_seir").html((parseFloat(data_H[data_H.length-1]).toFixed(2)).toString())
  $("#num_Of_criticals_seir").html((parseFloat(data_C[data_C.length-1]).toFixed(2)).toString())
  $("#num_Of_fatalities_seir").html((parseFloat(data_D[data_D.length-1]).toFixed(2)).toString())
  $("#num_Of_day_seir").html(((parseFloat(data_day[data_day.length-1])+1).toFixed(0)).toString())

  $("#num_Of_day_wm").html((($value_time_wm.val()-1)*30).toString())
  $("#num_Of_day_sd").html((($value_time_sd.val()-1)*30).toString())
  $("#num_Of_day_cs").html((($value_time_cs.val()-1)*30).toString())
  $("#num_Of_day_ld").html((($value_time_ld.val()-1)*30).toString())


}



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


function positives_draw(){
  if($id_switch_positive.checked == true){
    return data_num_positive;
  }
  else{
    return [];
  }
}
// function hospitalized_draw(){
//   if($id_switch_hospitalized.checked == true){
//       return data_H;
//   }
//   else{
//     return [];
//   }
// }
function cum_hospitalized_draw(){
  if($id_switch_cum_hospitalized.checked == true){
    return data_num_cumulative_hospitalizations;
  }
  else{
    return [];
  }
}

function criticals_draw(){
  if($id_switch_Criticals.checked == true){
    return data_C;
  }
  else{
    return [];
  }
}

function fatalities_draw(){
  if($id_switch_Death.checked == true){
  // if( true){
    return data_D;
  }
  else{
    return [];
  }
}
function susceptible_draw(){
  if($id_switch_Susceptible.checked == true){
    return data_S;
  }
  else{
    return [];
  }
}
function exposed_draw(){
  if($id_switch_Exosed.checked == true){
    return data_E;
  }
  else{
    return [];
  }
}

function infected_draw(){
  if($id_switch_Infected.checked == true){
    return data_I;
  }
  else{
    return [];
  }
}




function recovered_draw(){
  if($id_switch_Recovered.checked == true){
    return data_R;
  }
  else{
    return [];
  }
}
function hospitalized_seir_draw(){
  if($id_switch_Hospitalized.checked == true){
      return data_H;
    }
  else{
    return [];
  }
}
function target_hospitalized_draw(){
  if($id_switch_target_data.checked == true && $id_switch_Hospitalized.checked == true){
    return data_prof_num_hospitalised;
  }
  else{
    return [];
  }
}
function criticals_seir_draw(){
  if($id_switch_Criticals.checked == true){
      return data_C;
    }
  else{
    return [];
  }
}
function target_criticals_draw(){
  if($id_switch_target_data.checked == true && $id_switch_Criticals.checked == true){
    return data_prof_num_critical;
  }
  else{
    return [];
  }
}
function fatalities_seir_draw(){
  if($id_switch_Death.checked == true){
    return data_D;
  }
  else{
    return [];
  }
}
function target_death_draw(){
  if($id_switch_target_data.checked == true && $id_switch_Death.checked == true){
    return data_prof_num_fatalities;
  }
  else{
    return [];
  }
}



// $value_time_data.on('input change', () => {
//
//   $value_time_period_data.html($value_time_data.val());
//   load_cur_Data();
//   load_card_value();
// });


function load_card_kpi_value(){

  var percentChange_positive = data_num_positive[data_num_positive.length-1]-data_num_positive[data_num_positive.length-2]
  var percentChange_hos = data_num_hospitalised[data_num_hospitalised.length-1]-data_num_positive[data_num_hospitalised.length-2]
  var percentChange_cum_hos = data_num_hospitalised[data_num_cumulative_hospitalizations.length-1]-data_num_positive[data_num_cumulative_hospitalizations.length-2]
  var percentChange_crit = data_num_critical[data_num_critical.length-1]-data_num_critical[data_num_critical.length-2]
  var percentChange_fat = data_num_fatalities[data_num_fatalities.length-1]-data_num_fatalities[data_num_fatalities.length-2]

  if(percentChange_positive == 0){
    result = `
    <td class="fas fa-caret-right text-success">${ percentChange_positive }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#casesKpi").html(result)
  }
  else if(percentChange_positive > 0){
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_positive }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#casesKpi").html(result)
  }
  else{
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_positive }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#casesKpi").html(result)
  }


  if(percentChange_hos == 0){
    result = `
    <td class="fas fa-caret-right text-success">${ percentChange_hos }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_hospitalisedKpi").html(result)
  }
  else if(percentChange_hos < 0){
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_hos }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_hospitalisedKpi").html(result)
  }
  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_hos }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_hospitalisedKpi").html(result)
  }

    if(percentChange_cum_hos == 0){
      result = `
      <td class="fas fa-caret-right text-success">${ percentChange_cum_hos }</td>
      <td class="fas fa-percent text-success"></td>`
      $("#num_cmulative_hospitalizationsKpi").html(result)
    }
    else if(percentChange_cum_hos < 0){
      result = `
      <td class="fas fa-caret-down fa-1x text-success">${ percentChange_cum_hos }</td>
      <td class="fas fa-percent text-success"></td>`
      $("#num_cmulative_hospitalizationsKpi").html(result)
    }
    else{
      result = `
      <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_cum_hos }</td>
      <td class="fas fa-percent text-danger"></td>`
      $("#num_cmulative_hospitalizationsKpi").html(result)
    }

  if(percentChange_crit == 0){
    result = `
    <td class="fas fa-caret-right fa-1x text-success">${ percentChange_crit }</td>
    <td class="fas fa-percent text-success"></td>`

    $("#num_criticalKpi").html(result)
  }
  else if(percentChange_crit < 0){
    result = `


    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_crit }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_criticalKpi").html(result)
  }

  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_crit }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_criticalKpi").html(result)
  }
  if(percentChange_fat == 0){
    result = `
    <td class="fas fa-caret-right fa-1x text-success">${ percentChange_fat }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_fatalitiesKpi").html(result)
  }
  else if(percentChange_fat < 0){
    result = `
    <td class="fas fa-caret-down fa-1x text-success">${ percentChange_fat }</td>
    <td class="fas fa-percent text-success"></td>`
    $("#num_fatalitiesKpi").html(result)
  }

  else{
    result = `
    <td class="fas fa-caret-up fa-1x text-danger">${ percentChange_fat  }</td>
    <td class="fas fa-percent text-danger"></td>`
    $("#num_fatalitiesKpi").html(result)
  }




}


function draw() {

  if (typeof(myLineChart) != "undefined"){

    myLineChart.destroy();

  }


  ctx_active_cases = document.getElementById("myAreaSeirModel").getContext("2d");;
  myLineChart = new Chart(ctx_active_cases, {
    type: 'line',

    data: {
      labels: data_day,
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
          data: susceptible_draw(),
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
          data: exposed_draw(),
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
          data: infected_draw(),
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
          data: recovered_draw(),
        },
        // Hospitalized
        {
          label: "hospitalized ",
          lineTension: 0.6,
          backgroundColor: "rgba(34,139,34, 0.2)",
          borderColor: "rgba(34,139,34, 0.1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(34,139,34, 0.1)",
          pointBorderColor: "rgba(34,139,34, 0.1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(34,139,34, 0.1)",
          pointHoverBorderColor: "rgba(34,139,34, 0.1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: hospitalized_seir_draw(),
        },

        // Criticals
        {
          label: "Criticals ",
          lineTension: 0.6,
          backgroundColor: "rgba(255, 193, 7,0.3)",
          borderColor: "rgba(255, 193, 7,1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(255, 193, 7,1)",
          pointBorderColor: "rgba(255, 193, 7,1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(255, 193, 7, 1)",
          pointHoverBorderColor: "rgba(255, 193, 7, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: criticals_seir_draw(),
        },
        // Fatalities
        {
          label: "Fatalities",
          lineTension: 0.6,
          backgroundColor: "rgba(255, 193, 7,0.1)",
          borderColor: "rgba(237, 0, 59, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(237, 0, 59, 1)",
          pointBorderColor: "rgba(237, 0, 59, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(237, 0, 59, 1)",
          pointHoverBorderColor: "rgba(237, 0, 59, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: fatalities_seir_draw(),
        },
        // Plot data prof
        {
          label: "data_prof_num_hospitalised",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0,0.1)",
          borderColor: "rgba(0, 0, 0, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(0, 0, 0, 1)",
          pointBorderColor: "rgba(0, 0, 0, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
          pointHoverBorderColor: "rgba(0, 0, 0, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: target_hospitalized_draw(),

        },
        {
          label: "data_prof_num_critical",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0,0.1)",
          borderColor: "rgba(0, 0, 0, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(0, 0, 0, 1)",
          pointBorderColor: "rgba(0, 0, 0, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
          pointHoverBorderColor: "rgba(0, 0, 0, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: target_criticals_draw(),
        },
        {
          label: "data_prof_num_fatalities",
          lineTension: 0.6,
          backgroundColor: "rgba(0, 0, 0,0.1)",
          borderColor: "rgba(0, 0, 0, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(0, 0, 0, 1)",
          pointBorderColor: "rgba(0, 0, 0, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
          pointHoverBorderColor: "rgba(0, 0, 0, 1)",
          pointHitRadius: 5,
          pointBorderWidth: 4,
          data: target_death_draw(),
        },
        {
          label: "data_prof_num_fatalities",
          lineTension: 0.1,
          backgroundColor: "rgba(0, 0, 0,0.1)",
          borderColor: "rgba(0, 0, 0, 1)",
          pointRadius: 1,
          pointBackgroundColor: "rgba(0, 0, 0, 1)",
          pointBorderColor: "rgba(0, 0, 0, 1)",
          pointHoverRadius: 1,
          pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
          pointHoverBorderColor: "rgba(0, 0, 0, 1)",
          pointHitRadius: 1,
          pointBorderWidth: 1,
          data: data_ladder,
        }

        // // Plot scenario 2
        // {
        //   label: "data_prof_num_hospitalised",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_2_H,
        // },
        // {
        //   label: "data_prof_num_critical",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_2_C,
        // },
        // {
        //   label: "data_prof_num_fatalities",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_2_D,
        // },
        //
        // // Plot scenario 3
        // {
        //   label: "data_prof_num_hospitalised",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_3_H,
        // },
        // {
        //   label: "data_prof_num_critical",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_3_C,
        // },
        // {
        //   label: "data_prof_num_fatalities",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_3_D,
        // },
        // // Plot scenario 4
        // {
        //   label: "data_prof_num_hospitalised",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_4_H,
        // },
        // {
        //   label: "data_prof_num_critical",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_4_C,
        // },
        // {
        //   label: "data_prof_num_fatalities",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_4_D,
        // },
        // // Plot scenario 5
        // {
        //   label: "data_prof_num_hospitalised",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_5_H,
        // },
        // {
        //   label: "data_prof_num_critical",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_5_C,
        // },
        // {
        //   label: "data_prof_num_fatalities",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_5_D,
        // },
        // // Plot scenario 6
        // {
        //   label: "data_prof_num_hospitalised",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_6_H,
        // },
        // {
        //   label: "data_prof_num_critical",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_6_C,
        // },
        // {
        //   label: "data_prof_num_fatalities",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_6_D,
        // },
        // // Plot scenario 3
        // {
        //   label: "data_prof_num_hospitalised",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_7_H,
        // },
        // {
        //   label: "data_prof_num_critical",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_7_C,
        // },
        // {
        //   label: "data_prof_num_fatalities",
        //   lineTension: 0.6,
        //   backgroundColor: "rgba(0, 0, 0,0.1)",
        //   borderColor: "rgba(0, 0, 0, 1)",
        //   pointRadius: 1,
        //   pointBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHoverRadius: 1,
        //   pointHoverBackgroundColor: "rgba(0, 0, 0, 1)",
        //   pointHoverBorderColor: "rgba(0, 0, 0, 1)",
        //   pointHitRadius: 5,
        //   pointBorderWidth: 4,
        //   data: data_7_D,
        // },















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
