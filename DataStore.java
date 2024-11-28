
import java.util.*;
import weka.core.Instances;
import java.io.File;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
public class DataStore {
    private List<String> gameResults;
    private List<String> teamStatistics;
    private HashMap<String, List<TeamRecord>> teamDataMap;
    private Queue<String> gameResultQueue;
    private PriorityQueue<TeamData> teamRanking;
    private int[][] headToHead;
    private HashMap<String, Integer> teamIndexMap;
    public DataStore(int numTeams) {
        this.gameResults = new ArrayList<>();
        this.teamStatistics = new ArrayList<>();
        this.teamDataMap = new HashMap<>();
        this.gameResultQueue = new LinkedList<>();
        this.teamRanking = new PriorityQueue<>(Comparator.comparingDouble(TeamData::getSeedingScore).reversed());
        this.headToHead = new int[numTeams][numTeams];
        this.teamIndexMap = new HashMap<>();
    }
    public void addTeamToIndexMap(String team, int index){
        this.teamIndexMap.putIfAbsent(team, index);
    }
    public String[] getTeamsList(){
        String[] teamsList = new String[getNumTeams()];
        int i = 0;
        for (String team : teamIndexMap.keySet()){
            teamsList[i] = team;
            i++;
        }
        return teamsList;
    }
      public int getTeamIndex(String teamName) {
          return this.teamIndexMap.getOrDefault(teamName, -1);
      }
    public void addGameResult(String result) {
        gameResults.add(result);
        gameResultQueue.offer(result);  
    }
    public void addTeamStatistic(String statistic) {
        teamStatistics.add(statistic);
    }
    public void addTeamRecord(String teamName, TeamRecord record) {
        teamDataMap.computeIfAbsent(teamName, k -> new ArrayList<>()).add(record);
    }
public void updateHeadToHead(int teamIndex1, int teamIndex2, int result) {
    if (teamIndex1 < 0 || teamIndex2 < 0 || teamIndex1 >= headToHead.length || teamIndex2 >= headToHead.length) {
        System.out.println("Invalid team indices: " + teamIndex1 + ", " + teamIndex2);
        return; 
    }
    headToHead[teamIndex1][teamIndex2] += result;
}
    public int getHeadToHeadResult(int teamIndex1, int teamIndex2) {
        if (teamIndex1 < 0 || teamIndex2 < 0 ||
            teamIndex1 >= headToHead.length || teamIndex2 >= headToHead.length) {
            return 0; 
        }
        return headToHead[teamIndex1][teamIndex2];
    }
    public int getNumTeams() {
        return headToHead.length;
    }
    public void displayTeamData(String teamName) {
        List<TeamRecord> records = teamDataMap.get(teamName);
        if (records != null) {
            System.out.println("Data for team: " + teamName);
            for (TeamRecord record : records) {
                System.out.println(record);
            }
        } else {
            System.out.println("No data found for team: " + teamName);
        }
    }
    public void displayRankedTeams() {
        System.out.println("Team Rankings based on Seeding Score:");
        PriorityQueue<TeamData> tempRanking = new PriorityQueue<>(teamRanking);
        while (!tempRanking.isEmpty()) {
            System.out.println(tempRanking.poll());
        }
    }
    public static class TeamRecord {
        private int rank;
        private double statistic;
        private int year;
        public TeamRecord(int rank, double statistic, int year) {
            this.rank = rank;
            this.statistic = statistic;
            this.year = year;
        }
        @Override
        public String toString() {
            return "Year: " + year + ", Rank: " + rank + ", Statistic: " + statistic;
        }
    }
    public static class TeamData {
        private String teamName;
        private double seedingScore;  
        private int pointsScored;
        private int wins;
        private int losses;
        public TeamData(String teamName, double seedingScore, int pointsScored, int wins, int losses) {
            this.teamName = teamName;
            this.seedingScore = seedingScore;
            this.pointsScored = pointsScored;
            this.wins = wins;
            this.losses = losses;
        }
        public double getSeedingScore() {
            return seedingScore;
        }
        @Override
        public String toString() {
            return "Team: " + teamName + ", Seeding Score: " + seedingScore +
                   ", Points: " + pointsScored + ", Wins: " + wins + ", Losses: " + losses;
        }
    }
    public static void main(String[] args) {
        DataStore dataStore = new DataStore(30);
        try {
            String dataFolderPath = "/content/Cleaned_Data";  
            File dataFolder = new File(dataFolderPath);
            if (!dataFolder.exists() || !dataFolder.isDirectory()) {
                System.out.println("Directory not found: " + dataFolderPath);
                return;
            }
            for (File categoryFolder : dataFolder.listFiles()) {
                if (categoryFolder.isDirectory()) {
                    System.out.println("Category: " + categoryFolder.getName());
                    for (File teamFile : categoryFolder.listFiles()) {
                        if (teamFile.isFile() && teamFile.getName().endsWith(".csv")) {
                            DataSource source = new DataSource(teamFile.getPath());
                            Instances data = source.getDataSet();
                            String teamName = teamFile.getName().replace(".csv", "");
                            for (int i = 0; i < data.numInstances(); i++) {
                                Instance instance = data.instance(i);
                                int rank = (int) instance.value(0); 
                                double statistic = instance.value(1); 
                                int year = (int) instance.value(2); 
                                DataStore.TeamRecord record = new DataStore.TeamRecord(rank, statistic, year);
                                dataStore.addTeamRecord(teamName, record);
                            }
                        }
                    }
                }
            }
            dataStore.displayTeamData("Phoenix");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
