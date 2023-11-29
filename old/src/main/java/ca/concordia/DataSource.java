package ca.concordia;

import java.util.HashMap;
import java.util.Map;

public class DataSource {

    private String rootPath;

    public DataSource(String rootPath) {
        this.rootPath = rootPath;
    }

    public Map<String, String> getIMDBDatasets(){
        Map<String, String> tblName = new HashMap<>();
        tblName.put("Name", rootPath+"/ImdbName.csv");
        tblName.put("TitleCrew", rootPath+"/ImdbTitleCrew.csv");
        tblName.put("TitleRatings", rootPath+"/ImdbTitleRatings.csv");
        tblName.put("TitleAkas", rootPath+"/ImdbTitleAkas.csv");
        tblName.put("TitleEpisode", rootPath+"/ImdbTitleEpisode.csv");
        tblName.put("TitleBasics", rootPath+"/ImdbTitleBasics.csv");
        tblName.put("TitlePrincipals", rootPath+"/ImdbTitlePrincipals.csv");

        return tblName;
    }

    public Map<String, String> getYELPDatasets(){
        Map<String, String> tblName = new HashMap<>();
        tblName.put("business", rootPath+"/yelp_academic_dataset_business.json");
        tblName.put("review", rootPath+"/yelp_academic_dataset_review.json");
        tblName.put("user", rootPath+"/yelp_academic_dataset_user.json");
        tblName.put("checkin", rootPath+"/yelp_academic_dataset_checkin.json");
        tblName.put("tip", rootPath+"/yelp_academic_dataset_tip.json");
        tblName.put("photo", rootPath+"/photos.json");

        return tblName;
    }
}
