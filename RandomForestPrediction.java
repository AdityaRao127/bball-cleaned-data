import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import java.util.Scanner;
import java.io.*;
import java.util.*;
public class RandomForestPrediction {
    private static DataStore dataStore;
    private static FileWriter predictionWriter;
    private static FileWriter winLossWriter;
    public static Instances loadTrainingData(String cleanedDataPath) throws IOException {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("WeightedStat"));
        attributes.add(new Attribute("WinPercentage")); 
        Instances trainingData = new Instances("TrainingData", attributes, 100);
        trainingData.setClassIndex(1); 
        File cleanedDataDir = new File(cleanedDataPath);
        int teamIndex = 0;
        for (File statFolder : Objects.requireNonNull(cleanedDataDir.listFiles())) {
            if (statFolder.isDirectory()) {
                for (File teamFile : Objects.requireNonNull(statFolder.listFiles())) {
                  if (teamIndex < 30){
                        String team = teamFile.getName().split("\\.")[0];
                        dataStore.addTeamToIndexMap(team, teamIndex);
                        teamIndex++;
                    }
                    BufferedReader reader = new BufferedReader(new FileReader(teamFile));
                    String line = reader.readLine(); 
                    while ((line = reader.readLine()) != null) {
                        String[] columns = line.split(",");
                        double weightedStat = Double.parseDouble(columns[1].trim());
                        double winPercentage = Double.parseDouble(columns[3].trim());
                        Instance instance = new DenseInstance(2);
                        instance.setValue(attributes.get(0), weightedStat);
                        instance.setValue(attributes.get(1), winPercentage);
                        trainingData.add(instance);
                    }
                    reader.close();
                }
            }
        }
        return trainingData;
    }
public static void predictOutcomes(String teamName, String schedulePath, String opponentDataPath, RandomForest model) throws Exception {
    if (predictionWriter == null) {
        predictionWriter = new FileWriter("prediction_results.csv");
        predictionWriter.append("Team,Opponent,HSS Home,HSS Away,Win%\n"); 
    }
    if (winLossWriter == null) {
        winLossWriter = new FileWriter("win_loss_records.csv");
        winLossWriter.append("Team,Wins,Losses,HSS\n"); 
    }
    File scheduleFile = new File(schedulePath);
    BufferedReader reader = new BufferedReader(new FileReader(scheduleFile));
    List<Game> games = new ArrayList<>();
    String line = reader.readLine(); 
    while ((line = reader.readLine()) != null) {
        String[] columns = parseCSVLine(line);
        String date = columns[0].trim();
        String opponent = columns[2].trim();
        String location = columns[3].trim();
        String[] dateParts = date.split(",");
        int year = Integer.parseInt(dateParts[2].trim());
        games.add(new Game(date, opponent, location, year));
    }
    reader.close();
    ArrayList<Attribute> attributes = new ArrayList<>();
    attributes.add(new Attribute("WeightedStat"));
    Instances predictionData = new Instances("PredictionData", attributes, games.size());
    predictionData.setClassIndex(0);
    int winCount = 0;
    int lossCount = 0;
    double hssCount = 0;
for (int i = 0; i < games.size(); i++) {
    Game game = games.get(i);
    double homeHSS = loadHSS(teamName, opponentDataPath, game.year);
    hssCount += homeHSS;
    double awayHSS = loadHSS(game.opponent, opponentDataPath, game.year);
    if (game.location.equals("H")) {
        double homeAdvantageBoost = Math.max(0.5, awayHSS * 0.0325);
        homeHSS += homeAdvantageBoost; 
    }
    double weightedStat = homeHSS - awayHSS;
    Instance instance = new DenseInstance(1);
    instance.setValue(attributes.get(0), weightedStat);
    instance.setDataset(predictionData);
    double winPercentage = model.classifyInstance(instance);
    String predictedWinner;
    if (winPercentage > 0.5) {
        predictedWinner = teamName;
    } else if (winPercentage < 0.5) {
        predictedWinner = game.opponent;
    } else {
        predictedWinner = game.location.equals("H") ? teamName : game.opponent;
    }
    game.setResult(predictedWinner);
    dataStore.addGameResult(String.format("Game #%d: %s vs %s, Winner: %s", i + 1, teamName, game.opponent, predictedWinner));
    predictionWriter.append(String.format("%s,%s,%.5f,%.5f,%.5f\n", teamName, game.opponent, homeHSS, awayHSS, (100 * winPercentage)));
    int homeTeamIndex = getTeamIndex(teamName);
    int awayTeamIndex = getTeamIndex(game.opponent);
    if (predictedWinner.equals(teamName)) {
        dataStore.updateHeadToHead(homeTeamIndex, awayTeamIndex, 1);
        dataStore.updateHeadToHead(awayTeamIndex, homeTeamIndex, 0);
        winCount++; 
    } else {
        dataStore.updateHeadToHead(homeTeamIndex, awayTeamIndex, 0);
        dataStore.updateHeadToHead(awayTeamIndex, homeTeamIndex, 1);
        lossCount++;
    }
    printOutcomes(i + 1, teamName, game.opponent, homeHSS, awayHSS, winPercentage * 100, predictedWinner);
}
winLossWriter.append(String.format("%s,%d,%d,%.5f\n", teamName, winCount, lossCount, hssCount / games.size()));
}
private static int getTeamIndex(String teamName) {
    return dataStore.getTeamIndex(teamName);
}
private static double loadHSS(String team, String dataPath, int year) throws IOException {
    File currentDataDir = new File("Current_Data");
    File historicalDataDir = new File(dataPath);
    double totalStat = 0.0; 
    int statCount = 0; 
    for (File statFolder : Objects.requireNonNull(currentDataDir.listFiles())) {
        if (statFolder.isDirectory()) { 
            File teamFile = new File(statFolder, team + ".csv");
            if (teamFile.exists()) {
                BufferedReader reader = new BufferedReader(new FileReader(teamFile));
                String line = reader.readLine(); 
                if ((line = reader.readLine()) != null) {
                    String[] columns = line.split(",");
                    double statValue = Double.parseDouble(columns[1].trim()); 
                    totalStat += statValue;
                    statCount++;
                }
                reader.close();
            }
        }
    }
    if (statCount == 0) {
        for (File statFolder : Objects.requireNonNull(historicalDataDir.listFiles())) {
            if (statFolder.isDirectory()) { 
                File teamFile = new File(statFolder, team + ".csv");
                if (teamFile.exists()) {
                    BufferedReader reader = new BufferedReader(new FileReader(teamFile));
                    String line = reader.readLine(); 
                    while ((line = reader.readLine()) != null) {
                        String[] columns = line.split(",");
                        int teamYear = Integer.parseInt(columns[2].trim()); 
                        if (teamYear == year) {
                            double statValue = Double.parseDouble(columns[1].trim()); 
                            totalStat += statValue;
                            statCount++;
                        }
                    }
                    reader.close();
                }
            }
        }
    }
    if (statCount > 0) {
        double hss = totalStat;
        System.out.println("HSS for team: " + team + ", Year: " + year + " = " + hss);
        return hss;
    } else {
        System.out.println("No stats found for team: " + team + ", Year: " + year);
        return 0.0;
    }
}
    private static String[] parseCSVLine(String line) {
        List<String> columns = new ArrayList<>();
        boolean insideQuote = false;
        StringBuilder current = new StringBuilder();
        for (char c : line.toCharArray()) {
            if (c == '"') {
                insideQuote = !insideQuote; 
            } else if (c == ',' && !insideQuote) {
                columns.add(current.toString().trim());
                current.setLength(0); 
            } else {
                current.append(c);
            }
        }
        columns.add(current.toString().trim()); 
        return columns.toArray(new String[0]);
    }
private static void printOutcomes(int gameNumber, String homeTeam, String awayTeam, double homeHSS, double awayHSS, double predictedWinPercentage, String predictedWinner) {
    final String BOLD = "\033[1m";
    final String RESET = "\033[0m";
    System.out.printf(BOLD + "GAME #%d: " + RESET + "%s vs %s\n", gameNumber, homeTeam, awayTeam);
    System.out.printf("HSS %s: %.5f\n", homeTeam, homeHSS);
    System.out.printf("HSS %s: %.5f\n", awayTeam, awayHSS);
    System.out.printf("Predicted Win Percentage for %s: %.5f%%\n\n", homeTeam, predictedWinPercentage);
}
public static void queryWinsAndLosses(DataStore dataStore) { 
    Scanner scanner = new Scanner(System.in);
    System.out.println("Enter the name of the first team:");
    String team1 = scanner.nextLine().trim();
    System.out.println("Enter the name of the second team:");
    String team2 = scanner.nextLine().trim();
    int team1Index = dataStore.getTeamIndex(team1);
    int team2Index = dataStore.getTeamIndex(team2);
    if (team1Index == -1 || team2Index == -1) {
        System.out.println("One or both team names are invalid.");
        return;
    }
    int winsAgainst = dataStore.getHeadToHeadResult(team1Index, team2Index);
    int lossesAgainst = dataStore.getHeadToHeadResult(team2Index, team1Index);
    System.out.println(team1 + " vs " + team2 + ":");
    System.out.println(team1 + " wins: " + winsAgainst);
    System.out.println(team1 + " losses: " + lossesAgainst);
}
    public static void main(String[] args) {
        try {
            dataStore = new DataStore(30);
            String cleanedDataPath = "Cleaned_Data";
            String schedulePath = "Schedule/";
            String opponentDataPath = "Cleaned_Data";
            Instances trainingData = loadTrainingData(cleanedDataPath);
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100);
            rf.buildClassifier(trainingData);
            String[] teamsList = dataStore.getTeamsList();
            int numTeams = dataStore.getNumTeams();
            for (String team : teamsList) {
                predictOutcomes(team, schedulePath + team + "/" + team + ".csv", opponentDataPath, rf);
            }
            if (predictionWriter != null) {
                predictionWriter.flush();
                predictionWriter.close();
            }
            if (winLossWriter != null) {
                winLossWriter.flush();
                winLossWriter.close();
            }
            BufferedReader br = new BufferedReader(new FileReader(new File("win_loss_records.csv")));
            String line = br.readLine();
            while ((line = br.readLine()) != null) {
                String[] columns = line.split(",");
                String team = columns[0].trim();
                String wins = columns[1].trim();
                String losses = columns[2].trim();
                String hss = columns[3].trim();
                System.out.printf("%s had a predicted record of %s-%s and a HoopSight Strength Score of %s\n", team, wins, losses, hss);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
class Game {
    String date;
    String opponent;
    String location;
    int year;
    String result;
    public Game(String date, String opponent, String location, int year) {
        this.date = date;
        this.opponent = opponent;
        this.location = location;
        this.year = year;
    }
    public void setResult(String result) {
        this.result = result;
    }
}
