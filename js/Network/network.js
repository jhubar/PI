anychart.onDocumentReady(function () {
        anychart.data.loadJsonFile("https://static.anychart.com/git-storage/word-press/data/network-graph-tutorial/data_images.json", function (data) {

          // create a chart from the loaded data
          var chart = anychart.graph(data);

          // set the title
          chart.title("Network Graph showing the battles in Game of Thrones");

          // access nodes
          var nodes = chart.nodes();

          // set the size of nodes
          nodes.normal().height(30);
          nodes.hovered().height(45);
          nodes.selected().height(45);

          // set the stroke of nodes
          nodes.normal().stroke(null);
          nodes.hovered().stroke("#333333", 3);
          nodes.selected().stroke("#333333", 3);

          // enable the labels of nodes
          chart.nodes().labels().enabled(true);

          // configure the labels of nodes
          chart.nodes().labels().format("{%id}");
          chart.nodes().labels().fontSize(12);
          chart.nodes().labels().fontWeight(600);

          // draw the chart
          chart.container("container1").draw();

        });
      });
